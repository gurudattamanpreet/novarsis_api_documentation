from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import json
import time
import random
import requests
from datetime import datetime, timedelta
import base64
import io
from PIL import Image
import math
import logging
import hashlib
import html
import uvicorn
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Pydantic models for API requests and responses
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime
    show_feedback: bool = True


class ChatRequest(BaseModel):
    message: str
    image_data: Optional[str] = None
    platform: str = "mobile"
    device_info: Optional[Dict] = None


class FeedbackRequest(BaseModel):
    feedback: str
    message_index: int


class ChatResponse(BaseModel):
    response: str
    show_feedback: bool
    response_type: str
    quick_actions: List[Dict[str, str]]
    timestamp: str


class FeedbackResponse(BaseModel):
    response: str


class UploadResponse(BaseModel):
    image_data: str
    filename: str
    content_type: str
    instructions: str


class ChatHistoryResponse(BaseModel):
    chat_history: List[Dict]


class MobileChatResponse(BaseModel):
    status: str
    data: Dict


class MobileSuggestionsResponse(BaseModel):
    status: str
    data: Dict


class QuickActionResponse(BaseModel):
    status: str
    data: Dict


class SuggestionsResponse(BaseModel):
    suggestions: List[str]


class TypingSuggestionsResponse(BaseModel):
    suggestions: List[str]


# Initialize FastAPI app with additional metadata for documentation
app = FastAPI(
    title="Novarsis Support Center API",
    description="API for Novarsis SEO Tool AI Support Assistant",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json"  # OpenAPI schema
)


# Custom OpenAPI schema generation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/novarsis-logo.png"
    }

    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "Novarsis Support",
        "email": "support@novarsistech.com",
        "url": "https://novarsistech.com"
    }

    # Add license information
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }

    # Add servers information
    openapi_schema["servers"] = [
        {
            "url": "https://api.novarsistech.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.novarsistech.com",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Configure Ollama API - UPDATED FOR HOSTED SERVICE
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY",
                           "14bfe5365cc246dc82d933e3af2aa5b6.hz2asqgJi2bO_gpN7Cp1Hcku")  # Empty default, will be set via environment
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")  # Default to hosted service
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")  # Default model
USE_HOSTED_OLLAMA = True  # Always use hosted service

# Initialize Ollama model
model = True  # We'll assume it's available and handle errors in the API call

# Initialize embedding model - Ollama doesn't have a direct embedding API like Gemini
# So we'll use keyword-based filtering only
reference_embedding = None
embedding_model = None
logger.info("Using keyword-based filtering (Ollama doesn't provide embedding API)")

# Constants
WHATSAPP_NUMBER = "+91-9999999999"
SUPPORT_EMAIL = "support@novarsistech.com"

# Enhanced System Prompt - MOBILE APP OPTIMIZED
SYSTEM_PROMPT = """You are Nova, an AI assistant for Novarsis SEO Tool. Your role is to help users with SEO analysis, reports, account issues, and technical support.

IMPORTANT EMAIL RULES:
- NEVER ask users for their email address
- If a user voluntarily provides their email, acknowledge it properly
- Do not request email for connecting with experts or support
- Simply provide the support email (support@novarsistech.com) when needed
- Never say things like "Could you share your email address?" or "Please provide your email"

IMAGE ANALYSIS CAPABILITIES:
When a user attaches an image containing SEO errors or issues:
1. Analyze the screenshot/image for SEO-related errors
2. Identify specific error messages, codes, or issues shown
3. Provide clear explanations for each error
4. Suggest step-by-step solutions to fix the errors
5. Common SEO errors to look for:
   - Missing meta tags
   - Title/description length issues
   - H1/heading structure problems
   - Alt text missing
   - Broken links (404 errors)
   - Redirect chains
   - Duplicate content warnings
   - Page speed issues
   - Mobile optimization errors
   - Schema markup errors
   - Canonical URL issues
   - XML sitemap errors
   - Robots.txt issues
   - SSL certificate errors
   - Core Web Vitals issues

When analyzing error images:
- Be specific about what error you see
- Explain what the error means in simple terms
- Provide actionable solutions

IMPORTANT: You are responding in a MOBILE APP environment. Keep responses:
- SHORT and CONCISE (max 2-3 paragraphs)
- Use mobile-friendly formatting (short lines, clear breaks)
- Avoid long lists - use maximum 3-4 bullet points
- Use emojis sparingly for better mobile UX (✓ ✗ 📱 💡 ⚠️)
- Responses should fit on mobile screen without excessive scrolling

PERSONALITY:
- Natural and conversational like a human
- Friendly and approachable
- Brief but complete responses for mobile screens
- Polite and professional
- Ensure proper grammar with correct spacing and punctuation

INTRO RESPONSES:
- Who are you? → "I'm Nova, your personal assistant for Novarsis SEO Tool. I help users with SEO analysis, reports, account issues, and technical support."
- How can you help? → "I can help you with SEO website analysis, generating reports, fixing errors, managing subscriptions, and troubleshooting any Novarsis tool issues."
- What can you do? → "I assist with all Novarsis features - SEO audits, competitor analysis, keyword tracking, technical issues, billing, and more."

SCOPE:
Answer ALL questions naturally, but stay within Novarsis context:
• Greetings → Respond naturally (Hello! How can I help you today?)
• About yourself → Explain your role as Novarsis assistant
• Capabilities → List what you can help with
• Tool features → Explain Novarsis features
• Technical help → Provide solutions
• Account/billing → Assist with subscriptions

ONLY REDIRECT for completely unrelated topics like:
- Cooking recipes, travel advice, general knowledge
- Non-SEO tools or competitors
- Personal advice unrelated to SEO

For unrelated queries, politely say:
"Sorry, I only help with Novarsis SEO Tool.
Please let me know if you have any SEO tool related questions?"

RESPONSE STYLE (MOBILE OPTIMIZED):
- Natural conversation flow
- Keep responses SHORT for mobile screens
- 1-2 lines for simple queries, max 3-4 lines for complex ones
- Use simple, everyday language
- Break long sentences into shorter ones for mobile readability
- Use line breaks between different points
- Always use proper grammar with spaces between words and correct punctuation
- When user greets with a problem (e.g., "hi, what are features?"), skip greeting and answer directly
- Only greet back when user sends ONLY a greeting (like just "hi" or "hello")

CONTACT INFORMATION:
- When user says 'No' to "Have I resolved your query?", provide contact details:
  Contact Us:
  support@novarsistech.com
- Never use the phrase "For more information, please contact us on support@novarsistech.com"
- IMPORTANT: Always write emails correctly without spaces. The support email is: support@novarsistech.com (no spaces)
- When acknowledging user emails, write them correctly without spaces and preserve exact format (e.g., user@gmail.com not user@gmail. com or user@gmail. Com)
- CRITICAL EMAIL FORMAT: Always write emails as username@domain.com (all lowercase .com, no spaces)
- NEVER write emails as username@domain. Com (space and capital C is wrong)
- NEVER capitalize domain extensions (.Com, .Net, .Org are wrong - use .com, .net, .org)
- NEVER concatenate words with email addresses (e.g., "emailwdsjkd@gmail.com" is wrong, should be "wdsjkd@gmail.com")
- Always preserve the exact email format provided by user
- CRITICAL: When user provides an email like "wdsjkd@gmail.com", acknowledge it EXACTLY as "wdsjkd@gmail.com" - do NOT change it to "emaild@gmail.com" or any other variation
- When mentioning user's email in response, use the EXACT email they provided without any modifications
- Example: If user says "wdsjkd@gmail.com please check my account", respond with "Thanks for sharing your email wdsjkd@gmail.com" (not "emaild@gmail.com")

WEBSITE/DOMAIN FORMATTING RULES:
- CRITICAL: Always write domain names correctly without spaces (example.com, not example. com)
- NEVER write domains with spaces before extensions (example. Com is WRONG - use example.com)
- NEVER capitalize domain extensions (.Com is wrong - use .com)
- When user provides a website like "example.com", always refer to it EXACTLY as "example.com"
- Never add spaces in domain names: website.com ✓, website. com ✗, website.Com ✗
- Preserve exact domain formatting from user input
- IMPORTANT: When instructing to enter a website, write it as "enter example.com" NOT "enter example. Com"
- Always double-check domain formatting before sending response
- Examples of CORRECT formatting:
  * "enter example.com and tap Start"
  * "add example.com to the audit"
  * "visit website.org for more info"
- Examples of WRONG formatting:
  * "enter example. Com" (space before extension)
  * "add example.Com" (capital extension)
  * "visit website . org" (spaces around dot)

SPECIAL INSTRUCTIONS:
1. If user asks to connect with an expert or specialist:
   - DO NOT ask for their email address
   - Simply respond: "I'll forward your request to our SEO experts. They'll review your query and reach out through the appropriate channel."
   - Or provide: "Our experts can help you. Please contact: support@novarsistech.com" (NOT support@support@novarsistech.com)
   - NEVER write the email as support@support@ - always write it as support@novarsistech.com
   - NEVER say "Could you share your email address?" or similar
2. If the user asks for SEO analysis of a website, do not perform the analysis. Instead, guide them on how to do it in the Novarsis tool and provide general troubleshooting steps if they face issues.
3. IMPORTANT: When user asks about features of the tool, ONLY list the features. DO NOT mention pricing plans unless specifically asked about pricing, plans, or costs. Features include:
   ✓ Site audits & issue detection
   ✓ Keyword research & tracking
   ✓ Competitor analysis
   ✓ Backlink monitoring
   ✓ On-page SEO tips
   ✓ Rank tracking
   ✓ Custom reports
   ✓ Mobile optimization
4. When comparing pricing plans (ONLY when asked about pricing/plans/costs), use MOBILE-FRIENDLY format:

Free Plan
 5 websites
• All SEO tools
• 0/month

Pro Plan
50 websites
• Priority support
• 49/month

Enterprise
• Unlimited sites
• Dedicated manager
• Custom pricing

NEVER format plans in a single line like "Free Plan • 5 websites • All SEO tools"
ALWAYS use proper line breaks between plan name and features

5. If the user mentions multiple problems, address each one in your response.
6. At the end of your response, if you feel the answer might be incomplete or the user might need more help, ask: "Have I resolved your query?" If the user says no, then provide contact information:
   Contact Us:
   support@novarsistech.com
7. IMPORTANT: Never ask more than one question in a single response. This means:
   - If you have already asked a question (like an offer to contact support), do not ask 'Have I resolved your query?' in the same response.
   - If you are going to ask 'Have I resolved your query?', do not ask any other question in the same response.
8. If the user provides an email address, acknowledge it and continue the conversation. Do not restart the chat.
9. GREETING RULES:
   - If user says ONLY "hi", "hello", "hey" (single greeting), respond with: "Hello! I'm Nova, your personal assistant. How can I help you today?"
   - If user says greeting + problem (e.g., "hi, what are the features?"), SKIP the greeting and directly address the problem
   - Never start with a greeting when the user has already asked a question with their greeting
10. Never use the phrase "For more information, please contact us on" - instead just provide the email when needed as "Contact Us: support@novarsistech.com"
11. IMPORTANT: When you indicate that the issue is being handled by the team (e.g., "Our team will review", "get back to you", "working on your issue"), do NOT ask "Have I resolved your query?" because the issue is not yet resolved.
12. When asked about features, NEVER include pricing information unless explicitly asked. Only list the tool's features.
13. When a user provides an email address voluntarily, you may acknowledge it, but NEVER ask for email addresses.
"""

# Context-based quick reply suggestions
QUICK_REPLY_SUGGESTIONS = {
    "initial": [
        "How do I analyze my website SEO?",
        "Check my subscription status",
        "I'm getting an error message",
        "Generate SEO report",
        "Compare pricing plans"
    ],
    "seo_analysis": [
        "How to improve my SEO score?",
        "What are meta tags?",
        "Check page load speed",
        "Analyze competitor websites",
        "Mobile optimization tips"
    ],
    "account": [
        "Upgrade my plan",
        "Reset my password",
        "View billing history",
        "Cancel subscription",
        "Update payment method"
    ],
    "technical": [
        "API integration help",
        "Report not generating",
        "Login issues",
        "Data sync problems",
        "Browser compatibility"
    ],
    "report": [
        "Schedule automatic reports",
        "Export to PDF",
        "Share report with team",
        "Customize report sections",
        "Historical data comparison"
    ],
    "error": [
        "Website not loading",
        "Analysis stuck at 0%",
        "404 error on dashboard",
        "Payment failed",
        "Can't access reports"
    ],
    "pricing": [
        "What's included in Premium?",
        "Student discount available?",
        "Annual vs monthly billing",
        "Team plans pricing",
        "Free trial details"
    ]
}


# FAST MCP - Fast Adaptive Semantic Transfer with Memory Context Protocol
class FastMCP:
    def __init__(self):
        self.conversation_memory = []  # Full conversation memory
        self.context_window = []  # Recent context (last 10 messages)
        self.user_intent = None  # Current user intent
        self.topic_stack = []  # Stack of conversation topics
        self.entities = {}  # Named entities extracted
        self.user_profile = {
            "name": None,
            "plan": None,
            "issues_faced": [],
            "preferred_style": "concise",
            "interaction_count": 0
        }
        self.conversation_state = {
            "expecting_response": None,  # What type of response we're expecting
            "last_question": None,  # Last question asked by bot
            "pending_action": None,  # Any pending action
            "emotional_tone": "neutral"  # User's emotional state
        }

    def update_context(self, role, message):
        """Update conversation context with new message"""
        entry = {
            "role": role,
            "content": message,
            "timestamp": datetime.now(),
            "intent": self.extract_intent(message) if role == "user" else None
        }

        self.conversation_memory.append(entry)
        self.context_window.append(entry)

        # Keep context window to last 10 messages
        if len(self.context_window) > 10:
            self.context_window.pop(0)

        if role == "user":
            self.analyze_user_message(message)
        else:
            self.analyze_bot_response(message)

    def extract_intent(self, message):
        """Extract user intent from message"""
        message_lower = message.lower()

        # Intent patterns
        if any(word in message_lower for word in ['how', 'what', 'where', 'when', 'why']):
            return "question"
        elif any(word in message_lower for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'yep', 'yup']):
            return "confirmation"
        elif any(word in message_lower for word in ['no', 'nope', 'nah', 'not']):
            return "denial"
        elif any(word in message_lower for word in ['help', 'assist', 'support']):
            return "help_request"
        elif any(word in message_lower for word in ['error', 'issue', 'problem', 'broken', 'not working']):
            return "problem_report"
        elif any(word in message_lower for word in ['thanks', 'thank you', 'appreciate']):
            return "gratitude"
        elif any(word in message_lower for word in ['more', 'elaborate', 'explain', 'detail']):
            return "elaboration_request"
        else:
            return "statement"

    def analyze_user_message(self, message):
        """Analyze user message for context and emotion"""
        message_lower = message.lower()

        # Update emotional tone
        if any(word in message_lower for word in ['urgent', 'asap', 'immediately', 'quickly']):
            self.conversation_state["emotional_tone"] = "urgent"
        elif any(word in message_lower for word in ['frustrated', 'annoyed', 'angry', 'upset']):
            self.conversation_state["emotional_tone"] = "frustrated"
        elif any(word in message_lower for word in ['please', 'thanks', 'appreciate']):
            self.conversation_state["emotional_tone"] = "polite"

        # Extract entities
        if 'website' in message_lower or 'site' in message_lower:
            self.entities['subject'] = 'website'
        if 'seo' in message_lower:
            self.entities['subject'] = 'seo'
        if 'report' in message_lower:
            self.entities['subject'] = 'report'

        self.user_profile["interaction_count"] += 1

    def analyze_bot_response(self, message):
        """Track what the bot asked or offered"""
        message_lower = message.lower()

        if '?' in message:
            self.conversation_state["last_question"] = message
            self.conversation_state["expecting_response"] = "answer"

        if 'need more help' in message_lower or 'need help' in message_lower:
            self.conversation_state["expecting_response"] = "help_confirmation"

        if 'try these steps' in message_lower or 'follow these' in message_lower:
            self.conversation_state["expecting_response"] = "feedback_on_solution"

    def get_context_prompt(self):
        """Generate context-aware prompt for AI"""
        context_parts = []

        # Add conversation history
        if self.context_window:
            context_parts.append("=== Conversation Context ===")
            for entry in self.context_window[-5:]:  # Last 5 messages
                role = "User" if entry["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {entry['content']}")

        # Add conversation state
        if self.conversation_state["expecting_response"]:
            context_parts.append(f"\n[Expecting: {self.conversation_state['expecting_response']}]")

        if self.conversation_state["emotional_tone"] != "neutral":
            context_parts.append(f"[User tone: {self.conversation_state['emotional_tone']}]")

        if self.entities:
            context_parts.append(f"[Current topic: {', '.join(self.entities.values())}]")

        return "\n".join(context_parts)

    def should_filter_novarsis(self, message):
        """Determine if Novarsis filter should be applied"""
        # Don't filter if we're expecting a response to our question
        if self.conversation_state["expecting_response"] in ["help_confirmation", "answer", "feedback_on_solution"]:
            return False

        # Don't filter for contextual responses
        intent = self.extract_intent(message)
        if intent in ["confirmation", "denial", "elaboration_request"]:
            return False

        return True


# Initialize FAST MCP
fast_mcp = FastMCP()

# Global session state (in a real app, you'd use Redis or a database)
session_state = {
    "chat_history": [],
    "current_plan": None,
    "current_query": {},
    "typing": False,
    "user_name": "User",
    "session_start": datetime.now(),
    "resolved_count": 0,
    "pending_input": None,
    "uploaded_file": None,
    "intro_given": False,
    "last_user_query": "",
    "fast_mcp": fast_mcp,  # Add FAST MCP to session
    "last_bot_message_ends_with_query_solved": False
}

# Initialize current plan
plans = [
    {"name": "STARTER", "price": "$100/Year", "validity": "Valid till: Dec 31, 2025",
     "features": ["5 Websites", "Monthly Reports", "Email Support"]},
    {"name": "PREMIUM", "price": "$150/Year", "validity": "Valid till: Dec 31, 2025",
     "features": ["Unlimited Websites", "Real-time Reports", "Priority Support", "API Access"]}
]
session_state["current_plan"] = random.choice(plans)


# Helper Functions
def generate_avatar_initial(name):
    return name[0].upper()


def format_time(timestamp):
    return timestamp.strftime("%I:%M %p")


def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def is_greeting(query: str) -> bool:
    query_lower = query.lower().strip()
    return any(greeting in query_lower for greeting in GREETING_KEYWORDS)


def is_casual_allowed(query: str) -> bool:
    """Check if it's a casual/intro question that should be allowed"""
    query_lower = query.lower().strip()
    return any(word in query_lower for word in CAUSAL_ALLOWED)


def is_clearly_unrelated(query: str) -> bool:
    """Check if query is clearly unrelated to our tool"""
    query_lower = query.lower().strip()
    return any(topic in query_lower for topic in UNRELATED_TOPICS)


def is_novarsis_related(query: str) -> bool:
    # First check if it's a casual/intro question - always allow these
    if is_casual_allowed(query):
        return True

    # Check if it's clearly unrelated - always filter these
    if is_clearly_unrelated(query):
        return False

    # Since Ollama doesn't have embedding API, we use keyword-based filtering
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in NOVARSIS_KEYWORDS)


def get_intro_response() -> str:
    # Check if it's mobile platform
    if session_state.get("platform") == "mobile":
        return "Hi! I'm Nova 👋\nHow can I help you today?"
    return "Hello! I'm Nova, your personal assistant. How can I help you today?"


def call_ollama_api(prompt: str, image_data: Optional[str] = None) -> str:
    """Call Ollama API with the prompt - supports both local and hosted Ollama with image analysis"""
    try:
        # Check if using hosted service with API key
        if OLLAMA_API_KEY and USE_HOSTED_OLLAMA:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OLLAMA_API_KEY}"
            }
        else:
            # Local Ollama doesn't need auth
            headers = {
                "Content-Type": "application/json"
            }

        # Try different API formats based on service type
        if USE_HOSTED_OLLAMA:
            # For hosted service with image data
            if image_data:
                # Include image in the message for vision-capable models
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }]
            else:
                messages = [{"role": "user", "content": prompt}]

            data = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.7
            }
            endpoint = f"{OLLAMA_BASE_URL}/v1/chat/completions"  # OpenAI compatible endpoint
        else:
            # Local Ollama format
            data = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
            endpoint = f"{OLLAMA_BASE_URL}/api/generate"

        # If there's image data for local Ollama, include it
        if image_data and not USE_HOSTED_OLLAMA:
            data["images"] = [image_data]

        logger.info(f"Calling Ollama API at: {endpoint}")

        # Make the API call with increased timeout
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=60  # 60 seconds timeout
        )

        logger.info(f"Ollama response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Different response formats for local vs hosted
            if USE_HOSTED_OLLAMA:
                # OpenAI compatible format
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "No response generated.")
                else:
                    return result.get("response", "No response generated.")
            else:
                # Local Ollama format
                return result.get("response", "I couldn't generate a response. Please try again.")
        else:
            # Handle specific error codes
            if response.status_code == 401:
                return "Authentication error: Invalid API key. Please check your Ollama API key."
            elif response.status_code == 404:
                return f"Model not found: The model '{OLLAMA_MODEL}' is not available. Please check the model name."
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"API Error ({response.status_code}). Please check if the service is available."

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        return f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Please check your internet connection and the service URL."
    except requests.exceptions.Timeout:
        logger.error("Ollama API timeout")
        return "Response timeout. The service is taking too long to respond. Please try a simpler query."
    except Exception as e:
        logger.error(f"Ollama API error: {str(e)}")
        return f"Error: {str(e)}. Please check the logs for details."


# Novarsis Keywords - expanded for better detection
NOVARSIS_KEYWORDS = [
    'novarsis', 'seo', 'website analysis', 'meta tags', 'page structure', 'link analysis',
    'seo check', 'seo report', 'subscription', 'account', 'billing', 'plan', 'premium',
    'starter', 'error', 'bug', 'issue', 'problem', 'not working', 'failed', 'crash',
    'login', 'password', 'analysis', 'report', 'dashboard', 'settings', 'integration',
    'google', 'api', 'website', 'url', 'scan', 'audit', 'optimization', 'mobile', 'speed',
    'performance', 'competitor', 'ranking', 'keywords', 'backlinks', 'technical seo',
    'canonical', 'schema', 'sitemap', 'robots.txt', 'crawl', 'index', 'search console',
    'analytics', 'traffic', 'organic', 'serp'
]

# Casual/intro keywords that should be allowed
CAUSAL_ALLOWED = [
    'hello', 'hi', 'hey', 'who are you', 'what are you', 'what can you do',
    'how can you help', 'help me', 'assist', 'support', 'thanks', 'thank you',
    'bye', 'goodbye', 'good morning', 'good afternoon', 'good evening',
    'yes', 'no', 'okay', 'ok', 'sure', 'please', 'sorry'
]

# Clearly unrelated topics that should be filtered
UNRELATED_TOPICS = [
    'recipe', 'cooking', 'food', 'biryani', 'pizza', 'travel', 'vacation',
    'movie', 'song', 'music', 'game', 'sports', 'cricket', 'football',
    'weather', 'politics', 'news', 'stock', 'crypto', 'bitcoin',
    'medical', 'doctor', 'medicine', 'disease', 'health'
]

# Greeting keywords
GREETING_KEYWORDS = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]

# Set up templates - with error handling
try:
    templates = Jinja2Templates(directory="templates")
    logger.info("Templates initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize templates: {str(e)}")


    # Create a simple fallback template renderer
    class SimpleTemplates:
        def __init__(self, directory):
            self.directory = directory

        def TemplateResponse(self, name, context):
            # Simple fallback - just return a basic HTML response
            return HTMLResponse(
                "<html><body><h1>Novarsis Support Center</h1><p>Template rendering failed. Please check server logs.</p></body></html>")


    templates = SimpleTemplates("templates")


def get_mobile_quick_actions(response: str) -> list:
    """Get mobile-optimized quick action buttons based on response."""
    actions = []

    if "support" in response.lower():
        actions.append({"text": "📞 Call Support", "action": "call_support"})

    if "report" in response.lower():
        actions.append({"text": "📊 View Report", "action": "view_report"})

    if "upgrade" in response.lower() or "plan" in response.lower():
        actions.append({"text": "⬆️ Upgrade Plan", "action": "upgrade_plan"})

    # Always include help option
    if len(actions) < 3:
        actions.append({"text": "💬 Chat More", "action": "continue_chat"})

    return actions[:3]  # Max 3 actions for mobile UI


def get_context_suggestions(message: str) -> list:
    """Get relevant quick reply suggestions based on user's input context - MOBILE OPTIMIZED."""
    # Don't show suggestions for very short input (less than 3 characters)
    if not message or len(message.strip()) < 3:
        return []

    message_lower = message.lower().strip()

    # Return empty if message is still too short after stripping
    if len(message_lower) < 3:
        return []

    # Check for keywords and return appropriate suggestions
    if any(
            word in message_lower for word in ['seo', 'analysis', 'analyze', 'score', 'optimization', 'meta', 'crawl']):
        return QUICK_REPLY_SUGGESTIONS["seo_analysis"][:3]  # Max 3 for mobile
    elif any(word in message_lower for word in
             ['account', 'subscription', 'plan', 'billing', 'payment', 'upgrade', 'cancel']):
        return QUICK_REPLY_SUGGESTIONS["account"][:3]
    elif any(word in message_lower for word in
             ['error', 'issue', 'problem', 'not working', 'failed', 'stuck', 'broken']):
        return QUICK_REPLY_SUGGESTIONS["error"][:3]
    elif any(word in message_lower for word in ['report', 'export', 'pdf', 'schedule', 'download']):
        return QUICK_REPLY_SUGGESTIONS["report"][:3]
    elif any(word in message_lower for word in ['api', 'integration', 'technical', 'login', 'password']):
        return QUICK_REPLY_SUGGESTIONS["technical"][:3]
    elif any(word in message_lower for word in ['price', 'pricing', 'cost', 'plan', 'cheap', 'expensive', 'free']):
        return QUICK_REPLY_SUGGESTIONS["pricing"][:3]
    elif any(word in message_lower for word in ['how', 'what', 'why', 'when', 'where']):
        # For question words, show initial helpful suggestions
        return QUICK_REPLY_SUGGESTIONS["initial"][:3]  # Max 3 for mobile
    else:
        return []


def remove_duplicate_pricing(text: str) -> str:
    """Remove duplicate pricing plan entries"""
    lines = text.split('\n')
    seen_plans = set()
    filtered_lines = []
    current_plan = None

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this is a plan header
        if 'free plan' in line_lower:
            if 'free plan' not in seen_plans:
                seen_plans.add('free plan')
                current_plan = 'free plan'
                filtered_lines.append(line)
            else:
                current_plan = None  # Skip duplicate plan
        elif 'pro plan' in line_lower:
            if 'pro plan' not in seen_plans:
                seen_plans.add('pro plan')
                current_plan = 'pro plan'
                filtered_lines.append(line)
            else:
                current_plan = None
        elif 'enterprise' in line_lower and 'plan' not in line_lower:
            if 'enterprise' not in seen_plans:
                seen_plans.add('enterprise')
                current_plan = 'enterprise'
                filtered_lines.append(line)
            else:
                current_plan = None
        elif current_plan is not None:
            # This line belongs to the current plan
            filtered_lines.append(line)
        elif not any(p in line_lower for p in ['free plan', 'pro plan', 'enterprise']):
            # This line is not part of any plan
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def format_pricing_plans(text: str) -> str:
    """Format pricing plans to ensure proper structure and line breaks"""
    # First remove any duplicates
    text = remove_duplicate_pricing(text)

    # Check if the text contains pricing plan information
    if any(plan in text.lower() for plan in ["free plan", "pro plan", "enterprise"]):
        # Define the exact format we want (NO ASTERISKS, NO DOLLAR SIGNS)
        pricing_format = {
            "Free Plan": [
                " 5 websites",
                "• All SEO tools",
                "• 0/month"
            ],
            "Pro Plan": [
                "50 websites",
                "• Priority support",
                "• 49/month"
            ],
            "Enterprise": [
                "• Unlimited sites",
                "• Dedicated manager",
                "• Custom pricing"
            ]
        }

        # Look for any variation of pricing plans and replace with correct format
        for plan_name, features in pricing_format.items():
            # Pattern to match any variation of this plan
            patterns = [
                # Pattern with inline features
                rf'{plan_name}[:\s]*[^\n]*(?:websites|sites)[^\n]*(?:tools|support|manager)[^\n]*(?:month|pricing)',
                # Pattern with bullets on same line
                rf'{plan_name}[:\s]*•[^\n]+',
                # Simple plan name
                rf'{plan_name}[:\s]*'
            ]

            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Replace with properly formatted version (NO ASTERISKS)
                    formatted_plan = f"\n\n{plan_name}\n" + "\n".join(features)
                    text = re.sub(pattern, formatted_plan, text, flags=re.IGNORECASE)
                    break

        # Clean up any remaining formatting issues (NO ASTERISKS)
        text = re.sub(r'(Free Plan:?)\s*([•\-])?\s*', r'\n\nFree Plan\n', text, flags=re.IGNORECASE)
        text = re.sub(r'(Pro Plan:?)\s*([•\-])?\s*', r'\n\nPro Plan\n', text, flags=re.IGNORECASE)
        text = re.sub(r'(Enterprise:?)\s*([•\-])?\s*', r'\n\nEnterprise\n', text, flags=re.IGNORECASE)

        # Fix bullet points in pricing - ensure they're on new lines
        text = re.sub(r'([^\n])•', r'\1\n• ', text)
        text = re.sub(r'([^\n])\s*\-\s*([A-Z])', r'\1\n- \2', text)

        # Fix pricing amounts that got merged
        text = re.sub(r'(\$?\d+)\s*(month|/month|per month)', r'\1/month', text, flags=re.IGNORECASE)
        text = re.sub(r'(\$?\d+)\s*(year|/year|per year)', r'\1/year', text, flags=re.IGNORECASE)

        # Ensure proper spacing after features
        text = re.sub(r'(websites|sites)([A-Z])', r'\1\n\2', text)
        text = re.sub(r'(tools|support|access|manager|pricing|integrations)([A-Z•\-])', r'\1\n\2', text)

        # Fix merged plan names with features
        text = re.sub(r'(month|year|pricing)\s*([A-Z][a-z]+\s+Plan)', r'\1\n\n\2', text)

        # Clean up multiple spaces
        text = re.sub(r' {2,}', ' ', text)

        # Ensure bullet points have proper spacing
        text = re.sub(r'•([^\s])', r'• \1', text)

    return text


def remove_duplicate_questions(text: str) -> str:
    """Remove duplicate questions to ensure only one question appears at the end"""

    # Remove the "For more information" phrase completely
    text = re.sub(r'For more information[,.]?\s*please contact us on\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'For more information[,.]?\s*contact us at\s*', '', text, flags=re.IGNORECASE)

    # Check for escalation/contact instructions and remove "Have I resolved your query?" if it appears
    escalation_patterns = [
        r"please contact us",
        r"contact us at",
        r"reach out to us"
    ]

    for pattern in escalation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            query_solved_pos = text.find("Have I resolved your query?")
            if query_solved_pos != -1:
                # Remove the "Have I resolved your query?" part
                text = text[:query_solved_pos].strip()
            break

    # Check for phrases indicating the issue is being handled by the team
    team_handling_patterns = [
        r"Our team will",
        r"get back to you",
        r"review your",
        r"working on your",
        r"expert will reach out",
        r"team has been notified",
        r"will contact you"
    ]

    for pattern in team_handling_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            query_solved_pos = text.find("Have I resolved your query?")
            if query_solved_pos != -1:
                # Remove the "Have I resolved your query?" part
                text = text[:query_solved_pos].strip()
            break

    return text


def clean_response(text: str) -> str:
    """Clean and format the response text"""
    # Format pricing plans if present
    text = format_pricing_plans(text)

    # Remove duplicate questions
    text = remove_duplicate_questions(text)

    return text


def fix_common_spacing_issues(text: str) -> str:
    """Fix common spacing and hyphenation issues in text"""

    # Pattern to add space between alphanumeric characters (but not for ticket numbers)
    # First, protect ticket numbers
    import re
    ticket_pattern = r'(NVS\d+)'
    protected_tickets = {}

    # Find and protect all ticket numbers
    for match in re.finditer(ticket_pattern, text):
        placeholder = f'__TICKET_{len(protected_tickets)}__'
        protected_tickets[placeholder] = match.group()
        text = text.replace(match.group(), placeholder)

    # Now fix spacing between numbers and letters (but not within protected areas)
    # Add space between number and letter (e.g., "50claude" -> "50 claude")
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    # Add space between letter and number (e.g., "apple4" -> "apple 4")
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)

    # Restore protected ticket numbers
    for placeholder, original in protected_tickets.items():
        text = text.replace(placeholder, original)

    # Common words that are often incorrectly combined
    spacing_fixes = [
        # Time-related
        (r'\b(next)(week|month|year|day|time)\b', r'\1 \2'),
        (r'\b(last)(week|month|year|day|time|night)\b', r'\1 \2'),
        (r'\b(this)(week|month|year|day|time|morning|afternoon|evening)\b', r'\1 \2'),

        # Common phrases
        (r'\b(can)(not)\b', r'\1not'),  # cannot should be one word
        (r'\b(any)(one|body|thing|where|time|way|how)\b', r'\1\2'),  # anyone, anybody, etc.
        (r'\b(some)(one|body|thing|where|time|times|what|how)\b', r'\1\2'),  # someone, somebody, etc.
        (r'\b(every)(one|body|thing|where|time|day)\b', r'\1\2'),  # everyone, everybody, etc.
        (r'\b(no)(one|body|thing|where)\b', r'\1 \2'),  # noone -> no one needs space

        # Tool-related
        (r'\b(web)(site|page|master|mail)\b', r'\1\2'),
        (r'\b(data)(base|set)\b', r'\1\2'),
        (r'\b(back)(up|end|link|links|ground)\b', r'\1\2'),
        (r'\b(key)(word|words|board)\b', r'\1\2'),
        (r'\b(user)(name|names)\b', r'\1\2'),
        (r'\b(pass)(word|words)\b', r'\1\2'),
        (r'\b(down)(load|loads|time)\b', r'\1\2'),
        (r'\b(up)(load|loads|date|dates|grade|time)\b', r'\1\2'),

        # Business/SEO terms
        (r'\b(on)(line|board|going)\b', r'\1\2'),
        (r'\b(off)(line|board|set)\b', r'\1\2'),
        (r'\b(over)(view|all|load|time)\b', r'\1\2'),
        (r'\b(under)(stand|standing|stood|line|score)\b', r'\1\2'),
        (r'\b(out)(put|come|reach|line|look)\b', r'\1\2'),
        (r'\b(in)(put|come|sight|line|bound)\b', r'\1\2'),

        # Common compound words that need space
        (r'\b(alot)\b', r'a lot'),
        (r'\b(atleast)\b', r'at least'),
        (r'\b(aswell)\b', r'as well'),
        (r'\b(inorder)\b', r'in order'),
        (r'\b(upto)\b', r'up to'),
        (r'\b(setup)\b', r'set up'),  # as verb

        # Fix "Im" -> "I'm"
        (r'\b(Im)\b', r"I'm"),
        (r'\b(Ive)\b', r"I've"),
        (r'\b(Ill)\b', r"I'll"),
        (r'\b(Id)\b', r"I'd"),
        (r'\b(wont)\b', r"won't"),
        (r'\b(cant)\b', r"can't"),
        (r'\b(dont)\b', r"don't"),
        (r'\b(doesnt)\b', r"doesn't"),
        (r'\b(didnt)\b', r"didn't"),
        (r'\b(isnt)\b', r"isn't"),
        (r'\b(arent)\b', r"aren't"),
        (r'\b(wasnt)\b', r"wasn't"),
        (r'\b(werent)\b', r"weren't"),
        (r'\b(hasnt)\b', r"hasn't"),
        (r'\b(havent)\b', r"haven't"),
        (r'\b(hadnt)\b', r"hadn't"),
        (r'\b(wouldnt)\b', r"wouldn't"),
        (r'\b(couldnt)\b', r"couldn't"),
        (r'\b(shouldnt)\b', r"shouldn't"),
        (r'\b(youre)\b', r"you're"),
        (r'\b(youve)\b', r"you've"),
        (r'\b(youll)\b', r"you'll"),
        (r'\b(youd)\b', r"you'd"),
        (r'\b(hes)\b', r"he's"),
        (r'\b(shes)\b', r"she's"),
        (r'\b(its)\b(?! \w+ing)', r"it's"),  # its -> it's (but not before -ing verbs)
        (r'\b(were)\b(?! \w+ing)', r"we're"),  # were -> we're contextually
        (r'\b(theyre)\b', r"they're"),
        (r'\b(theyve)\b', r"they've"),
        (r'\b(theyll)\b', r"they'll"),
        (r'\b(theyd)\b', r"they'd"),
        (r'\b(whats)\b', r"what's"),
        (r'\b(wheres)\b', r"where's"),
        (r'\b(theres)\b', r"there's"),
        (r'\b(thats)\b', r"that's"),

        # Common hyphenated words
        (r'\b(re)(check|restart|send|reset|do|run|build)\b', r'\1-\2'),
        (r'\b(pre)(view|set|defined|configured)\b', r'\1-\2'),
        (r'\b(co)(operate|ordinate|author)\b', r'\1-\2'),
        (r'\b(multi)(purpose|factor|level)\b', r'\1-\2'),
        (r'\b(self)(service|help|hosted)\b', r'\1-\2'),
        (r'\b(real)(time)\b', r'\1-\2'),
        (r'\b(up)(to)(date)\b', r'\1-\2-\3'),
        (r'\b(state)(of)(the)(art)\b', r'\1-\2-\3-\4'),

        # Fix spacing around punctuation
        (r'\s+([.,!?;:])', r'\1'),  # Remove space before punctuation
        (r'([.,!?;:])([A-Za-z])', r'\1 \2'),  # Add space after punctuation

        # Fix multiple spaces
        (r'\s+', r' '),
    ]

    # Apply all fixes
    for pattern, replacement in spacing_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Special case for "no one" (needs space)
    text = re.sub(r'\b(noone)\b', r'no one', text, flags=re.IGNORECASE)

    # Ensure proper capitalization at sentence start
    text = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), text)
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

    return text


def format_response_text(text: str) -> str:
    """Format the response text to ensure proper bullet points and numbered lists"""
    # Split text into lines for processing
    lines = text.split('\n')
    formatted_lines = []

    for line in lines:
        # Skip empty lines
        if not line.strip():
            formatted_lines.append('')
            continue

        # Process numbered lists (e.g., "1. ", "2. ", etc.)
        if re.match(r'^\s*\d+\.\s+', line):
            # This is a numbered list item, ensure it's on its own line
            formatted_lines.append(line)

        # Process bullet points (e.g., "- ", "• ", etc.)
        elif re.match(r'^\s*[-•]\s+', line):
            # This is a bullet point, ensure it's on its own line
            formatted_lines.append(line)

        # Check if line contains numbered list items in the middle
        elif re.search(r'\s\d+\.\s+', line):
            # Split the line at numbered list items
            parts = re.split(r'(\s\d+\.\s+)', line)
            new_line = parts[0]
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    # Add the numbered item on a new line
                    new_line += '\n' + parts[i] + parts[i + 1]
                else:
                    new_line += parts[i]
            formatted_lines.append(new_line)

        # Check if line contains bullet points in the middle
        elif re.search(r'\s[-•]\s+', line):
            # Split the line at bullet points
            parts = re.split(r'(\s[-•]\s+)', line)
            new_line = parts[0]
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    # Add the bullet point on a new line
                    new_line += '\n' + parts[i] + parts[i + 1]
                else:
                    new_line += parts[i]
            formatted_lines.append(new_line)

        # Regular text
        else:
            formatted_lines.append(line)

    # Join the formatted lines
    formatted_text = '\n'.join(formatted_lines)

    # Additional formatting for pricing plans
    if "Free Plan:" in formatted_text and "Pro Plan:" in formatted_text and "Enterprise Plan:" in formatted_text:
        # Extract the pricing section
        pricing_start = formatted_text.find("Free Plan:")
        if pricing_start != -1:
            # Find the end of the pricing section
            pricing_end = formatted_text.find("Would you like me to connect with an expert for the Enterprise model?")
            if pricing_end == -1:
                pricing_end = len(formatted_text)

            pricing_section = formatted_text[pricing_start:pricing_end]

            # Format each pricing plan
            plans = re.split(r'(Free Plan:|Pro Plan:|Enterprise Plan:)', pricing_section)
            formatted_plans = []

            for i in range(1, len(plans), 2):
                if i + 1 < len(plans):
                    plan_name = plans[i]
                    plan_details = plans[i + 1]

                    # Format the plan details with bullet points
                    details = plan_details.split('-')
                    formatted_details = [details[0].strip()]  # First part (e.g., "Up to 5 websites")

                    for detail in details[1:]:
                        if detail.strip():
                            formatted_details.append(f"- {detail.strip()}")

                    formatted_plans.append(f"{plan_name}\n" + '\n'.join(formatted_details))

            # Replace the pricing section in the original text
            formatted_text = formatted_text[:pricing_start] + '\n\n'.join(formatted_plans) + formatted_text[
                                                                                             pricing_end:]

    return formatted_text


def format_response_lists(text: str) -> str:
    """Format numbered lists and bullet points to appear on separate lines with proper spacing"""

    # First handle variations of "follow these steps" or similar phrases
    step_intros = [
        r'(follow these steps?:?)\s*',
        r'(here are the steps?:?)\s*',
        r'(try these steps?:?)\s*',
        r'(please try:?)\s*',
        r'(steps to follow:?)\s*',
        r'(you can:?)\s*',
        r'(to do this:?)\s*',
    ]

    for pattern in step_intros:
        text = re.sub(pattern + r'(\d+\.)', r'\1\n\n\2', text, flags=re.IGNORECASE)

    # Fix numbered lists that appear inline (e.g., "text. 1. item 2. item")
    # Add newline before numbers that follow a period but aren't already on new line
    text = re.sub(r'([.!?])\s+(\d+\.\s+)', r'\1\n\n\2', text)

    # Handle numbered items that are separated by just a space
    # Pattern: "1. something 2. something" -> "1. something\n2. something"
    text = re.sub(r'(\d+\.[^\n.!?]+[.!?]?)\s+(\d+\.\s+)', r'\1\n\n\2', text)

    # Ensure numbered items at start of line
    text = re.sub(r'(?<!\n)(\d+\.\s+[A-Z])', r'\n\1', text)

    # Handle bullet points (-, *, •)
    # Add newline before bullet if not already there
    text = re.sub(r'(?<!\n)\s*([•\-\*])\s+([A-Z])', r'\n\1 \2', text)

    # Handle "Plan details" and plan names
    text = re.sub(r'(Plan details?:?)\s*(?!\n)', r'\n\n\1\n', text, flags=re.IGNORECASE)

    # Format each plan name on new line with proper spacing
    plan_names = ['Free Plan:', 'Pro Plan:', 'Premium Plan:', 'Enterprise Plan:', 'Starter Plan:', 'Basic Plan:']
    for plan in plan_names:
        # Look for plan name and ensure it's on new line with spacing
        text = re.sub(r'(?<!\n)({plan})', r'\n\n\1', text, flags=re.IGNORECASE)
        # Add newline after plan name if features follow immediately
        text = re.sub(r'({plan})\s*([A-Z\-•])', r'\1\n\2', text, flags=re.IGNORECASE)

    # Handle Step-by-step instructions
    text = re.sub(r'(?<!\n)(Step\s+\d+[:.])\s*', r'\n\n\1 ', text, flags=re.IGNORECASE)

    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)

    # Clean up excessive newlines but keep proper spacing
    text = re.sub(r'\n{4,}', r'\n\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    return text.strip()


def format_response_presentable(text: str) -> str:
    """Make the response more presentable with proper formatting"""

    # Ensure questions are on new paragraphs
    questions_patterns = [
        r'(Would you like[^?]+\?)',
        r'(Do you [^?]+\?)',
        r'(Have I [^?]+\?)',
        r'(Should I [^?]+\?)',
        r'(Can I [^?]+\?)',
        r'(Shall I [^?]+\?)',
        r'(For more information[^?]+\?)',
        r'(Is there [^?]+\?)',
        r'(Did this [^?]+\?)',
        r'(Does this [^?]+\?)',
    ]

    for pattern in questions_patterns:
        # Add double newline before question if not already present
        text = re.sub(r'(?<!\n\n)' + pattern, r'\n\n\1', text, flags=re.IGNORECASE)

    # Format specific sections that often appear
    # Ticket information
    text = re.sub(r'(Ticket (?:Number|ID):\s*NVS\d+)', r'\n\1', text)

    # Format error/solution sections
    text = re.sub(r'((?:Error|Solution|Note|Tip|Warning|Important):)\s*', r'\n\n\1\n', text, flags=re.IGNORECASE)

    # Ensure proper paragraph breaks after sentences before certain keywords
    paragraph_triggers = [
        'To ', 'For ', 'Please ', 'You can ', 'Try ', 'Follow ',
        'First ', 'Second ', 'Third ', 'Next ', 'Then ', 'Finally ',
        'Additionally ', 'Also ', 'Furthermore ', 'However ',
    ]

    for trigger in paragraph_triggers:
        text = re.sub(rf'([.!?])\s+({trigger})', r'\1\n\n\2', text)

    # SPECIAL PRICING FORMATTING
    # Detect and format pricing sections
    if 'Plan' in text and any(word in text for word in ['websites', 'month', 'pricing', 'features']):
        # Ensure plan names are on new lines with proper spacing
        text = re.sub(r'(?<!\n\n)(Free Plan)', r'\n\n\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\n\n)(Pro Plan)', r'\n\n\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\n\n)(Enterprise Plan)', r'\n\n\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\n\n)(Premium Plan)', r'\n\n\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\n\n)(Starter Plan)', r'\n\n\1', text, flags=re.IGNORECASE)

        # Format bullet points properly
        text = re.sub(r'([^\n])•', r'\1\n•', text)  # Ensure bullet on new line
        text = re.sub(r'•\s*([^\n]+)\s*•', r'• \1\n•', text)  # Split merged bullets

    # Clean up spacing issues
    text = re.sub(r'\s*\n\s*', r'\n', text)  # Remove spaces around newlines
    text = re.sub(r'\n{4,}', r'\n\n', text)  # Max 2 newlines
    text = re.sub(r'^\n+', '', text)  # Remove leading newlines
    text = re.sub(r'\n+$', '', text)  # Remove trailing newlines

    return text


def fix_email_format(text: str) -> str:
    """Fix email formatting issues in the response - COMPREHENSIVE FIX"""

    # First, handle the double support@ issue specifically
    text = re.sub(r'support@support@novarsistech\.com', 'support@novarsistech.com', text, flags=re.IGNORECASE)
    text = re.sub(r'support@support@', 'support@', text, flags=re.IGNORECASE)

    # Then, handle all other variations
    # Using a more aggressive approach

    # Pattern to match all variations of the email
    # This will catch: supportnovarsistech. Com, supportnovarsistech.Com, etc.
    email_patterns = [
        # With or without @, with spaces around dot and Com/com
        r'support(?:@)?\s*novarsistech\s*\.\s*[Cc]om',
        # Without dot
        r'support(?:@)?\s*novarsistech\s+[Cc]om',
        # With multiple spaces
        r'support\s+novarsistech\s*\.?\s*[Cc]om',
        # Just the domain part when it appears alone
        r'novarsistech\s*\.\s*[Cc]om',
        # With tech separated
        r'support(?:@)?\s*novarsis\s*tech\s*\.?\s*[Cc]om',
    ]

    # Apply all patterns
    for pattern in email_patterns:
        text = re.sub(pattern, 'support@novarsistech.com', text, flags=re.IGNORECASE)

    # Special handling for when it appears in context
    # "contact us on/at" followed by any email variation
    text = re.sub(
        r'(contact\s+us\s+(?:on|at)\s+)[a-z]*novarsis[a-z]*\s*\.?\s*[Cc]om\.?',
        r'\1support@novarsistech.com',
        text,
        flags=re.IGNORECASE
    )

    # "email us at" followed by any email variation
    text = re.sub(
        r'(email\s+us\s+at\s+)[a-z]*novarsis[a-z]*\s*\.?\s*[Cc]om\.?',
        r'\1support@novarsistech.com',
        text,
        flags=re.IGNORECASE
    )

    # Handle if there's a period after .com (like ". Com.")
    text = re.sub(r'support@novarsistech\.com\.', 'support@novarsistech.com.', text)

    # Final cleanup - remove any remaining malformed emails
    # This is a catch-all for any we might have missed
    if 'novarsistech' in text.lower() and '@' not in text[
                                                     max(0, text.lower().find('novarsistech') - 10):text.lower().find(
                                                         'novarsistech') + 30]:
        # Find all occurrences and fix them
        matches = list(re.finditer(r'\b[a-z]*novarsis[a-z]*\s*\.?\s*[Cc]om\b', text, re.IGNORECASE))
        for match in reversed(matches):  # Process in reverse to maintain indices
            start, end = match.span()
            # Check if this looks like it should be an email
            before_text = text[max(0, start - 20):start].lower()
            if any(word in before_text for word in ['contact', 'email', 'at', 'on', 'us', 'support']):
                text = text[:start] + 'support@novarsistech.com' + text[end:]

    return text


def get_ai_response(user_input: str, image_data: Optional[str] = None, chat_history: list = None) -> str:
    try:
        # Get FAST MCP instance
        mcp = session_state.get("fast_mcp", FastMCP())

        # Update MCP with user input
        mcp.update_context("user", user_input)

        # Check if we should apply Novarsis filter
        should_filter = mcp.should_filter_novarsis(user_input)

        # Special handling for image attachments
        if image_data:
            # Check if the message is SEO-related or general error query
            seo_image_keywords = ['error', 'seo', 'issue', 'problem', 'bug', 'fix', 'help', 'analyze', 'screenshot',
                                  'tool', 'novarsis']
            is_seo_related_image = any(keyword in user_input.lower() for keyword in seo_image_keywords)

            # If no context provided or it's clearly not SEO-related, filter it
            if not is_seo_related_image and not any(keyword in user_input.lower() for keyword in NOVARSIS_KEYWORDS):
                return """Sorry, I only help with the Novarsis SEO Tool.

Please let me know if you have any SEO-related questions?"""

        # Only filter if MCP says we should
        elif should_filter and not is_novarsis_related(user_input):
            return """Sorry, I only help with Novarsis SEO Tool.

Please let me know if you have any SEO tool related questions?"""

        # Get context from MCP
        context = mcp.get_context_prompt()

        # Enhanced system prompt based on emotional tone
        enhanced_prompt = SYSTEM_PROMPT
        if mcp.conversation_state["emotional_tone"] == "urgent":
            enhanced_prompt += "\n[User is urgent - provide immediate, actionable solutions]"
        elif mcp.conversation_state["emotional_tone"] == "frustrated":
            enhanced_prompt += "\n[User is frustrated - be extra helpful and empathetic]"

        # Create the full prompt with special handling for images
        if image_data:
            # Enhanced prompt for image analysis
            image_analysis_prompt = """\n\nIMPORTANT: The user has attached an image containing SEO-related errors.
            Please analyze the image and:
            1. Identify all visible SEO errors or issues shown in the screenshot
            2. For each error, provide:
               - The exact error message or issue type
               - A clear explanation of what this error means
               - Step-by-step instructions to fix the error
            3. If multiple errors are visible, address each one separately
            4. Use simple, non-technical language where possible
            5. If you cannot identify specific SEO errors in the image, ask the user to describe what error they're experiencing

            Common SEO errors to look for in screenshots:
            - Meta tag issues (missing, too long, too short)
            - Heading structure problems (missing H1, multiple H1s)
            - Missing alt text on images
            - Page speed scores and issues
            - Mobile usability errors
            - 404 errors and broken links
            - SSL/HTTPS warnings
            - Schema markup errors
            - Core Web Vitals metrics
            - Duplicate content warnings

            Format your response clearly with the error type as a header, followed by explanation and solution."""

            prompt = f"{enhanced_prompt}{image_analysis_prompt}\n\n{context}\n\nUser query with SEO error screenshot: {user_input}\n\n[Analyze the attached image for SEO-related errors and provide detailed solutions]"
        else:
            prompt = f"{enhanced_prompt}\n\n{context}\n\nUser query: {user_input}"

        # Call Ollama API
        response_text = call_ollama_api(prompt, image_data)

        # Check if API returned an error
        if "Error:" in response_text or "cannot connect" in response_text.lower():
            logger.error(f"API Error in response: {response_text}")
            # Return a more helpful message instead of the raw error
            return "I'm having trouble connecting to the AI service right now. Please try again in a moment, or contact support@novarsistech.com for assistance."

        # Debug: Print the response before processing
        logger.info(f"Response received, length: {len(response_text)}")

        # ULTRA EARLY PRICING FIX - If this is a pricing query, replace the ENTIRE response
        if ('pricing' in user_input.lower() or 'plans' in user_input.lower() or
                'price' in user_input.lower() or 'cost' in user_input.lower()):
            # This is a pricing query - return the correct format immediately
            return """Free Plan
• 5 websites
• All SEO tools
• 0/month

Pro Plan
• 50 websites
• Priority support
• 49/month

Enterprise
• Unlimited sites
• Dedicated manager
• Custom pricing

Have I resolved your query?"""

        # ULTRA EARLY FIX: Fix domain spacing issues immediately after getting response
        # This pattern catches "domain. Com" or "domain . Com" etc.
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s*\.\s+([Cc][Oo][Mm])\b', r'\1.com', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s*\.\s+([Nn][Ee][Tt])\b', r'\1.net', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s*\.\s+([Oo][Rr][Gg])\b', r'\1.org', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s*\.\s+([Cc][Oo])\b', r'\1.co', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s*\.\s+([Ii][Oo])\b', r'\1.io', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s*\.\s+([Ii][Nn])\b', r'\1.in', response_text)

        # Fix capitalized extensions
        response_text = response_text.replace('. Com', '.com')
        response_text = response_text.replace('.Com', '.com')
        response_text = response_text.replace('. NET', '.net')
        response_text = response_text.replace('.NET', '.net')
        response_text = response_text.replace('. ORG', '.org')
        response_text = response_text.replace('.ORG', '.org')

        # Specific fix for the exact pattern you're seeing
        response_text = response_text.replace('example. Com', 'example.com')
        response_text = response_text.replace('example. com', 'example.com')
        response_text = response_text.replace('example .com', 'example.com')
        response_text = response_text.replace('example . com', 'example.com')

        # CRITICAL: Fix ALL domain names and URLs (not just emails)
        # Extract any URLs/domains from user input
        url_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,})'
        url_matches = re.findall(url_pattern, user_input, re.IGNORECASE)

        # Also look for simple domain patterns
        simple_domain_pattern = r'\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})\b'
        simple_matches = re.findall(simple_domain_pattern, user_input, re.IGNORECASE)
        url_matches.extend(simple_matches)

        # Fix each found domain in the response
        for domain in url_matches:
            # Clean the domain (lowercase, no spaces)
            clean_domain = domain.lower().strip()

            # Create all possible corrupted variations
            domain_parts = clean_domain.split('.')
            if len(domain_parts) >= 2:
                domain_name = '.'.join(domain_parts[:-1])  # Everything except TLD
                tld = domain_parts[-1]  # The TLD (com, net, org, etc.)

                # Fix variations with space and capitalization
                corrupted_domain_patterns = [
                    # Domain with space before dot: "domain . com" or "domain .com"
                    rf'{re.escape(domain_name)}\s+\.\s*{re.escape(tld)}',
                    rf'{re.escape(domain_name)}\s*\.\s+{re.escape(tld)}',
                    # Domain with capital TLD: "domain.Com"
                    rf'{re.escape(domain_name)}\.{re.escape(tld.capitalize())}',
                    # Domain with space and capital: "domain. Com" or "domain . Com"
                    rf'{re.escape(domain_name)}\s*\.\s*{re.escape(tld.capitalize())}',
                    # Any weird capitalization of the TLD
                    rf'{re.escape(domain_name)}\s*\.\s*{re.escape(tld.upper())}',
                    # Handle if domain name itself got capitalized
                    rf'{re.escape(domain_name.capitalize())}\s*\.\s*{re.escape(tld)}',
                    rf'{re.escape(domain_name.capitalize())}\s*\.\s*{re.escape(tld.capitalize())}',
                ]

                for pattern in corrupted_domain_patterns:
                    response_text = re.sub(pattern, clean_domain, response_text, flags=re.IGNORECASE)

        # Fix common domain extensions with spaces/capitals for ANY domain
        # This catches domains not in user input too
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s+\.\s*([Cc][Oo][Mm])\b', r'\1.com', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s+\.\s*([Nn][Ee][Tt])\b', r'\1.net', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s+\.\s*([Oo][Rr][Gg])\b', r'\1.org', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s+\.\s*([Ii][Oo])\b', r'\1.io', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s+\.\s*([Cc][Oo])\b', r'\1.co', response_text)

        # Fix domains ending with ". Com" (space + capital)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\s*\.\s*Com\b', r'\1.com', response_text)
        response_text = re.sub(r'([a-zA-Z0-9-]+)\.Com\b', r'\1.com', response_text)

        # Additional comprehensive domain fixes
        # Fix any domain pattern with space before TLD
        response_text = re.sub(r'([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*)\s+\.\s*([a-zA-Z]{2,})\b', r'\1.\2', response_text)
        # Fix capitalized TLDs
        response_text = re.sub(r'\.([A-Z]{2,})\b', lambda m: '.' + m.group(1).lower(), response_text)
        # Fix space after dot in domains
        response_text = re.sub(r'([a-zA-Z0-9-]+)\.\s+([a-zA-Z]{2,})\b', r'\1.\2', response_text)

        # CRITICAL EMAIL FIX: Extract and preserve user's email from input FIRST
        user_email = None
        user_email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', user_input)
        if user_email_match:
            user_email = user_email_match.group()
            logger.info(f"User provided email: {user_email}")

            # Find ANY mention of an email in response that looks like it could be the user's email
            # This includes truncated or corrupted versions
            domain = user_email.split('@')[1]  # e.g., "gmail.com"
            username = user_email.split('@')[0]  # e.g., "ehdhk"

            # Create patterns to match corrupted versions of the user's email
            corruption_patterns = [
                # Truncated username: "k@gmail.com" instead of "ehdhk@gmail.com"
                rf'\b[a-z]{{1,3}}@{re.escape(domain)}',
                # Space in domain: "ehdhk@gmail. com" or "k@gmail. Com"
                rf'[a-zA-Z0-9._%+-]*@{domain.split(".")[0]}\s*\.\s*{domain.split(".")[1]}',
                # Partial username with space in domain
                rf'{username[-3:] if len(username) > 3 else username}@{domain.split(".")[0]}\s*\.\s*[Cc]om',
                # Just the last letter(s): "k@gmail.Com" or "hk@gmail.com"
                rf'{username[-1]}@{re.escape(domain)}',
                rf'{username[-2:] if len(username) > 2 else username}@{re.escape(domain)}',
                # Any short variation with the domain
                rf'\b\w{{1,5}}@{re.escape(domain)}',
                # Domain with capitalization issues
                rf'[a-zA-Z0-9._%+-]*@{domain.split(".")[0]}\s*\.\s*[Cc]om',
                # The word "email" followed by truncated version
                rf'email\s+\w{{1,5}}@{re.escape(domain)}',
                # Any mention of partial username@domain
                rf'\b\w*{username[-1]}@{re.escape(domain)}',
            ]

            # Replace ALL corrupted versions with the correct email
            for pattern in corruption_patterns:
                matches = list(re.finditer(pattern, response_text, re.IGNORECASE))
                for match in matches:
                    # Check if this isn't the support email
                    if 'support' not in match.group().lower() and 'novarsis' not in match.group().lower():
                        logger.info(f"Replacing corrupted email: {match.group()} with {user_email}")
                        response_text = response_text[:match.start()] + user_email + response_text[match.end():]

            # ADDITIONAL FIX: Look for the exact user email with space/capitalization issues
            # This catches cases where the full email is present but formatted wrong
            # e.g., "ejdneajd@gmail. Com" -> "ejdneajd@gmail.com"
            corrupted_exact_patterns = [
                # Username with space after dot: "ejdneajd@gmail. com"
                rf'{re.escape(username)}@{domain.split(".")[0]}\s*\.\s*{domain.split(".")[1]}',
                # Username with capital Com: "ejdneajd@gmail.Com"
                rf'{re.escape(username)}@{domain.split(".")[0]}\.{domain.split(".")[1].capitalize()}',
                # Username with space and capital: "ejdneajd@gmail. Com"
                rf'{re.escape(username)}@{domain.split(".")[0]}\s*\.\s*{domain.split(".")[1].capitalize()}',
                # Any capitalization variation
                rf'{re.escape(username)}@{domain.split(".")[0]}\s*\.\s*[Cc][Oo][Mm]',
            ]

            for pattern in corrupted_exact_patterns:
                if re.search(pattern, response_text, re.IGNORECASE):
                    logger.info(f"Fixing exact email corruption: {pattern}")
                    response_text = re.sub(pattern, user_email, response_text, flags=re.IGNORECASE)

        # COMPREHENSIVE EMAIL FIXES for ALL emails (not just user's)
        # Fix patterns like "email. com" or "email. Com"
        response_text = re.sub(
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)\s+\.\s*([Cc][Oo][Mm]|[Cc]om|[Cc]o\.in|[Nn]et|[Oo]rg|[Ii]n|[Ii]o)',
            r'\1.com', response_text)

        # Fix any email ending with ". Com" (space + capital C)
        response_text = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)\s*\.\s*Com\b', r'\1.com', response_text)

        # Fix any email ending with ".Com" (no space, capital C)
        response_text = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)\.Com\b', r'\1.com', response_text)

        # Fix user emails that got corrupted (e.g., "gbgbnd@gmail. Com" -> "gbgbnd@gmail.com")
        response_text = re.sub(r'@([a-zA-Z0-9.-]+)\s+\.\s*([Cc][Oo][Mm]|[Cc]om)', r'@\1.com', response_text)
        response_text = re.sub(r'@([a-zA-Z0-9.-]+)\s+\.\s*([Nn][Ee][Tt]|[Nn]et)', r'@\1.net', response_text)
        response_text = re.sub(r'@([a-zA-Z0-9.-]+)\s+\.\s*([Oo][Rr][Gg]|[Oo]rg)', r'@\1.org', response_text)
        response_text = re.sub(r'@([a-zA-Z0-9.-]+)\s+\.\s*([Cc][Oo]\.in)', r'@\1.co.in', response_text)

        # Fix support@ duplication EARLY - Multiple patterns to catch all variations
        response_text = re.sub(r'support@support@novarsistech', 'support@novarsistech', response_text,
                               flags=re.IGNORECASE)
        response_text = re.sub(r'support@support@', 'support@', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'email:\s*support@support@', 'email: support@', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'contact:\s*support@support@', 'contact: support@', response_text, flags=re.IGNORECASE)

        # Fix the specific pattern: "you can email: support@support@..."
        response_text = re.sub(r'(you can email|can email|email them at|reach them at)\s*:\s*support@support@',
                               r'\1: support@', response_text, flags=re.IGNORECASE)

        # Fix Contact Us: support@support@ pattern
        response_text = re.sub(r'(Contact\s+Us\s*:\s*)support@support@', r'\1support@', response_text,
                               flags=re.IGNORECASE)

        # Fix any sentence ending with support@support@
        response_text = re.sub(r'(email|contact|reach|write to)\s*:\s*support@support@', r'\1: support@', response_text,
                               flags=re.IGNORECASE)

        # Fix patterns where space got inserted in domain (gmail. com -> gmail.com)
        response_text = re.sub(
            r'@(gmail|yahoo|hotmail|outlook|aol|icloud|proton|mail|email)\s*\.\s*([Cc]om|[Nn]et|[Oo]rg)', r'@\1.\2',
            response_text, flags=re.IGNORECASE)

        # If user_email exists, do a final pass to ensure it's correctly formatted everywhere
        if user_email:
            # Make sure user's email is properly formatted (fix any remaining issues)
            response_text = response_text.replace(user_email.replace('.com', '.Com'), user_email)
            response_text = response_text.replace(user_email.replace('.com', '. com'), user_email)
            response_text = response_text.replace(user_email.replace('.com', '. Com'), user_email)

        # Fix alphanumeric spacing (but protect emails)
        # Protect email addresses first - improved pattern to catch more variations
        protected_emails = []
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        for match in re.finditer(email_pattern, response_text):
            placeholder = f'__EMAIL_{len(protected_emails)}__'
            protected_emails.append(match.group())
            response_text = response_text.replace(match.group(), placeholder)

        # Also protect any remaining email-like patterns that might have been missed
        # This catches patterns like "email@domain.com" or "user@domain. com"
        extended_email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,}'
        for match in re.finditer(extended_email_pattern, response_text):
            if f'__EMAIL_{len(protected_emails)}__' not in response_text:  # Avoid duplicates
                placeholder = f'__EMAIL_{len(protected_emails)}__'
                protected_emails.append(match.group().replace(' ', ''))  # Clean the email
                response_text = response_text.replace(match.group(), placeholder)

        # Now add spaces between numbers and letters
        response_text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', response_text)
        response_text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', response_text)

        # Restore protected emails
        for i, email in enumerate(protected_emails):
            response_text = response_text.replace(f'__EMAIL_{i}__', email)

        # NOW fix email formatting after alphanumeric spacing
        response_text = fix_email_format(response_text)

        # Enhanced cleaning for grammar and formatting
        # Remove ALL asterisk symbols (both ** and single *)
        response_text = re.sub(r'\*+', '', response_text)  # Remove all asterisks
        response_text = response_text.replace("**", "")  # Extra safety for double asterisks
        # Remove any repetitive intro lines if present
        response_text = re.sub(r'^(Hey there[!,. ]*I\'?m Nova.*?assistant[.!]?\s*)', '', response_text,
                               flags=re.IGNORECASE).strip()
        # Keep alphanumeric, spaces, common punctuation, newlines, and bullet/section characters
        response_text = re.sub(r'[^a-zA-Z0-9 .,!?:;()\n•@-]', '', response_text)

        # Fix common grammar issues
        # Ensure space after period if not followed by a newline
        response_text = re.sub(r'\.([A-Za-z])', r'. \1', response_text)
        # Fix double spaces
        response_text = re.sub(r'\s+', ' ', response_text)
        # Ensure space after comma
        response_text = re.sub(r',([A-Za-z])', r', \1', response_text)
        # Ensure space after question mark and exclamation
        response_text = re.sub(r'([!?])([A-Za-z])', r'\1 \2', response_text)
        # Fix missing spaces between words
        response_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', response_text)

        # --- Formatting improvements for presentability ---
        # Normalize multiple spaces
        response_text = re.sub(r'\s+', ' ', response_text)
        # Ensure proper paragraph separation
        response_text = re.sub(r'([.!?])\s', r'\1\n\n', response_text)

        # CRITICAL PRICING FIX - Complete replacement for any pricing response
        # This should happen EARLY in the processing pipeline
        if ('pricing' in response_text.lower() or 'plans' in response_text.lower() or
                'free plan' in response_text.lower() or 'pro plan' in response_text.lower()):

            # If we detect ANY pricing-related content, replace EVERYTHING with the correct format
            if any(x in response_text.lower() for x in ['5 websites', '50 websites', 'unlimited sites',
                                                        'all seo tools', 'priority support', 'dedicated manager',
                                                        '0/month', '49/month', 'custom pricing']):
                # This is definitely a pricing response - replace it completely
                response_text = """Free Plan
• 5 websites
• All SEO tools
• 0/month

Pro Plan
• 50 websites
• Priority support
• 49/month

Enterprise
• Unlimited sites
• Dedicated manager
• Custom pricing

Have I resolved your query?"""

                # Skip all other formatting - return immediately
                return response_text.strip()

        # Format the response text to ensure proper bullet points and numbered lists
        response_text = format_response_text(response_text)

        # --- End formatting improvements ---

        # Clean the response (format pricing, remove duplicate questions, fix ticket numbers)
        response_text = clean_response(response_text)

        # SPECIAL PRICING FORMAT FIX
        # Fix the specific pattern you're seeing
        if any(plan in response_text for plan in ['Free Plan', 'Pro Plan', 'Enterprise']):
            # More comprehensive pattern to handle inline pricing plans
            # Pattern 1: "Free Plan 5 websites All SEO tools 0/month" (no bullets)
            response_text = re.sub(r'Free Plan\s+5\s+websites\s+All\s+SEO\s+tools\s+0/month',
                                   r'\n\nFree Plan\n 5 websites\n• All SEO tools\n• 0/month', response_text)
            response_text = re.sub(r'Pro Plan\s+50\s+websites\s+Priority\s+support\s+49/month',
                                   r'\n\nPro Plan\n50 websites\n• Priority support\n• 49/month', response_text)
            response_text = re.sub(r'Enterprise\s+Unlimited\s+sites\s+Dedicated\s+manager\s+Custom\s+pricing',
                                   r'\n\nEnterprise\n• Unlimited sites\n• Dedicated manager\n• Custom pricing',
                                   response_text)

            # Pattern 2: Fix patterns like "Free Plan • 5 websites • All SEO tools • 0 month"
            response_text = re.sub(r'\*\*Free Plan\*\*', 'Free Plan', response_text)  # Remove asterisks
            response_text = re.sub(r'\*\*Pro Plan\*\*', 'Pro Plan', response_text)
            response_text = re.sub(r'\*\*Enterprise\*\*', 'Enterprise', response_text)
            response_text = re.sub(r'Free Plan\s*•\s*([^\n•]+)', r'\n\nFree Plan\n 5 websites', response_text)
            response_text = re.sub(r'Pro Plan\s*•\s*([^\n•]+)', r'\n\nPro Plan\n50 websites', response_text)
            response_text = re.sub(r'Enterprise\s*•\s*([^\n•]+)', r'\n\nEnterprise\n• Unlimited sites', response_text)

            # Fix merged bullet points - more comprehensive
            response_text = re.sub(r'•\s*([^\n•]{1,50})\s*•', r'• \1\n•', response_text)

            # Split features that are merged into single line
            response_text = re.sub(r'( 5\s+websites)\s+([A-Z])', r'\1\n• \2', response_text)
            response_text = re.sub(r'(50\s+websites)\s+([^\n])', r'\1\n• \2', response_text)
            response_text = re.sub(r'(•\s+All\s+SEO\s+tools)\s+([^\n])', r'\1\n• \2', response_text)
            response_text = re.sub(r'(•\s+Priority\s+support)\s+([^\n])', r'\1\n• \2', response_text)
            response_text = re.sub(r'(•\s+Unlimited\s+sites)\s+([A-Z])', r'\1\n• \2', response_text)
            response_text = re.sub(r'(•\s+Dedicated\s+manager)\s+([A-Z])', r'\1\n• \2', response_text)

            # Fix pricing that got merged (like "0 month" should be "0/month")
            # NOTE: NO DOLLAR SIGNS as per requirement
            response_text = re.sub(r'(\d+)\s+month\b', r'\1/month', response_text)
            response_text = re.sub(r'(\d+)\s+year\b', r'\1/year', response_text)
            # Remove any dollar signs that might have been added
            response_text = re.sub(r'\$(\d+)/month', r'\1/month', response_text)
            response_text = re.sub(r'\$(\d+)/year', r'\1/year', response_text)

            # Ensure each bullet point is on new line
            lines = response_text.split('\n')
            formatted_lines = []
            for line in lines:
                if '•' in line:
                    # Split by bullet and format
                    parts = line.split('•')
                    if len(parts) > 1:
                        formatted_lines.append(parts[0].strip())
                        for part in parts[1:]:
                            if part.strip():
                                formatted_lines.append('• ' + part.strip())
                else:
                    formatted_lines.append(line)
            response_text = '\n'.join(formatted_lines)

        # Fix common spacing and grammar issues
        response_text = fix_common_spacing_issues(response_text)

        # Format numbered lists and bullet points for better presentation
        response_text = format_response_lists(response_text)

        # Make the response more presentable
        response_text = format_response_presentable(response_text)

        # Ensure "Have I resolved your query?" is always on a new paragraph
        if "Have I resolved your query?" in response_text:
            # Replace any occurrence where it's not after a newline
            response_text = response_text.replace(" Have I resolved your query?", "\n\nHave I resolved your query?")
            # Also handle if it's at the start of a line but without enough spacing
            response_text = response_text.replace("\nHave I resolved your query?", "\n\nHave I resolved your query?")
            # Clean up any triple newlines that might have been created
            response_text = re.sub(r'\n{3,}Have I resolved your query\?', '\n\nHave I resolved your query?',
                                   response_text)

        # FINAL EMAIL FIX - Run this at the very end to catch any corrupted emails
        # This is the last line of defense

        # CRITICAL: Final fix for support@support@ duplication
        response_text = re.sub(r'support@support@novarsistech\.com', 'support@novarsistech.com', response_text,
                               flags=re.IGNORECASE)
        response_text = re.sub(r'support@support@', 'support@', response_text, flags=re.IGNORECASE)

        # Fix standard support email variations
        response_text = re.sub(
            r'support(?:@)?\s*novarsis\s*tech\s*\.\s*[Cc]om',
            'support@novarsistech.com',
            response_text,
            flags=re.IGNORECASE
        )
        # Also fix variations without 'support'
        response_text = re.sub(
            r'(?:contact\s+us\s+(?:on|at)\s+)\s*novarsis\s*tech\s*\.\s*[Cc]om',
            'support@novarsistech.com',
            response_text,
            flags=re.IGNORECASE
        )

        # FINAL CLEANUP - Remove "For more information" phrase if it still exists
        response_text = re.sub(
            r'For more information[,.]?\s*please contact us on\s*',
            'Contact Us: ',
            response_text,
            flags=re.IGNORECASE
        )
        response_text = re.sub(
            r'For more information[,.]?\s*contact us at\s*',
            'Contact Us: ',
            response_text,
            flags=re.IGNORECASE
        )

        # FINAL PRICING CHECK - Ensure all three plans are shown
        if ('pricing' in response_text.lower() or 'plans' in response_text.lower() or
                ('free plan' in response_text.lower() and '5 websites' in response_text.lower())):
            # Count how many plans are mentioned
            plans_mentioned = []
            if 'free plan' in response_text.lower():
                plans_mentioned.append('free')
            if 'pro plan' in response_text.lower():
                plans_mentioned.append('pro')
            if 'enterprise' in response_text.lower():
                plans_mentioned.append('enterprise')

            # If not all three plans are mentioned, replace with complete pricing
            if len(plans_mentioned) < 3:
                complete_pricing = """Free Plan
• 5 websites
• All SEO tools
• 0/month

Pro Plan
• 50 websites
• Priority support
• 49/month

Enterprise
• Unlimited sites
• Dedicated manager
• Custom pricing"""

                # Preserve follow-up question if exists
                if "Have I resolved your query?" in response_text:
                    response_text = complete_pricing + "\n\nHave I resolved your query?"
                else:
                    response_text = complete_pricing

        # ABSOLUTE FINAL DOMAIN FIX - One more pass to catch any remaining issues
        # Extract domains from user input one more time for final check
        user_domains = re.findall(r'\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})\b', user_input, re.IGNORECASE)
        for domain in user_domains:
            clean_domain = domain.lower().strip()
            # Find and replace ANY variation of this domain
            domain_name = clean_domain.split('.')[0]
            domain_ext = clean_domain.split('.')[-1]

            # Create a super aggressive pattern that catches ANY variation
            # This will match: domain. com, domain .com, domain. Com, domain .Com, etc.
            super_pattern = rf'{re.escape(domain_name)}\s*\.\s*{re.escape(domain_ext)}'
            response_text = re.sub(super_pattern, clean_domain, response_text, flags=re.IGNORECASE)

            # Also fix if the extension got capitalized
            wrong_domain = f'{domain_name}.{domain_ext.capitalize()}'
            response_text = response_text.replace(wrong_domain, clean_domain)
            wrong_domain = f'{domain_name}. {domain_ext.capitalize()}'
            response_text = response_text.replace(wrong_domain, clean_domain)
            wrong_domain = f'{domain_name} . {domain_ext.capitalize()}'
            response_text = response_text.replace(wrong_domain, clean_domain)

        return response_text.strip()
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(
            f"User input was: {user_input[:100]}..." if len(user_input) > 100 else f"User input was: {user_input}")
        # Return a more helpful error message
        return "I'm experiencing a temporary issue. Please try your question again, or for immediate assistance, contact us at support@novarsistech.com"


# API Routes
@app.get("/", response_class=HTMLResponse, tags=["General"])
async def read_root(request: Request):
    """
    Render the main chat interface.

    Args:
        request (Request): The incoming request object.

    Returns:
        HTMLResponse: The rendered HTML template for the chat interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a user message and return AI assistant response.

    This endpoint handles both text and image inputs, providing SEO assistance
    through the Nova AI assistant.

    Args:
        request (ChatRequest): The chat request containing message and optional image.

    Returns:
        ChatResponse: The AI response with metadata.
    """
    # Check if request is from mobile
    is_mobile = request.platform == "mobile"

    # Special handling for image attachments
    if request.image_data:
        # Check if this is likely an SEO-related screenshot
        seo_keywords = ['error', 'seo', 'issue', 'problem', 'fix', 'help', 'analyze', 'tool', 'novarsis', 'website',
                        'meta', 'tag', 'speed', 'mobile']

        # If user hasn't provided context, check if it might be SEO-related
        if not request.message or request.message.strip() == "":
            request.message = "Please analyze this screenshot."
        elif len(request.message.strip()) < 20:
            # If message is too short, check if it contains SEO keywords
            if not any(keyword in request.message.lower() for keyword in seo_keywords):
                # Could be non-SEO screenshot, let the AI determine
                request.message = f"{request.message}. Please analyze this screenshot."
            else:
                # Likely SEO-related, enhance the message
                request.message = f"{request.message}. This screenshot shows SEO-related issues. Please help me understand and fix them."

    # Add mobile context to session if mobile
    if is_mobile:
        session_state["platform"] = "mobile"

    # Check if the user is responding to "Have I resolved your query?"
    if session_state.get("last_bot_message_ends_with_query_solved"):
        if request.message.lower() in ["no", "nope", "not really", "not yet"]:
            # User says no, so we provide contact information
            session_state["last_bot_message_ends_with_query_solved"] = False
            response = """Contact Us:
support@novarsistech.com"""
            bot_message = {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(),
                "show_feedback": True
            }
            session_state["chat_history"].append(bot_message)
            return {
                "response": response,
                "show_feedback": True,
                "response_type": "text",
                "quick_actions": [],
                "timestamp": datetime.now().isoformat()
            }
        elif request.message.lower() in ["yes", "yeah", "yep", "thank you", "thanks"]:
            # User says yes, we can acknowledge
            session_state["last_bot_message_ends_with_query_solved"] = False
            response = "Great! I'm glad I could help. Feel free to ask if you have any more questions about Novarsis! 🚀"
            bot_message = {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(),
                "show_feedback": True
            }
            session_state["chat_history"].append(bot_message)
            return {"response": response, "show_feedback": True}

    # Check if the message is an email
    if re.match(r"[^@]+@[^@]+\.[^@]+", request.message):
        # It's an email, so we acknowledge and continue
        # We don't want to restart the chat, so we just pass it to the AI
        pass  # We'll let the AI handle it as per the system prompt

    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now()
    }
    session_state["chat_history"].append(user_message)

    # Store current query for potential escalation
    session_state["current_query"] = {
        "query": request.message,
        "timestamp": datetime.now()
    }

    # Store last user query for "Connect with an Expert"
    session_state["last_user_query"] = request.message

    # Get AI response with chat history for context
    time.sleep(0.5)  # Simulate thinking time

    if is_greeting(request.message):
        # Check if there's more content after the greeting (like a problem)
        message_lower = request.message.lower()
        # Remove greeting words to check if there's additional content
        remaining_message = request.message
        for greeting in GREETING_KEYWORDS:
            if greeting in message_lower:
                # Remove the greeting word (case-insensitive) and common punctuation
                remaining_message = re.sub(rf'\b{greeting}\b[,.]?\s*', '', remaining_message, flags=re.IGNORECASE)
                break

        remaining_message = remaining_message.strip()

        # If there's content after greeting, handle the FULL MESSAGE but with instruction to skip greeting
        if remaining_message and len(remaining_message) > 2:
            # Pass the full message but with special instruction to skip greeting
            enhanced_input = f"[USER HAS GREETED WITH PROBLEM - SKIP GREETING AND DIRECTLY ADDRESS THE ISSUE]\n{request.message}"
            response = get_ai_response(enhanced_input, request.image_data, session_state["chat_history"])
        else:
            # Just greeting
            response = get_intro_response()

        session_state["intro_given"] = True
        show_feedback = True  # Changed to True
    else:
        response = get_ai_response(request.message, request.image_data, session_state["chat_history"])
        show_feedback = True  # Already True

    # Update FAST MCP with bot response
    if "fast_mcp" in session_state:
        session_state["fast_mcp"].update_context("assistant", response)

    # Check if the response ends with "Have I resolved your query?"
    if response.strip().endswith("Have I resolved your query?"):
        session_state["last_bot_message_ends_with_query_solved"] = True
    else:
        session_state["last_bot_message_ends_with_query_solved"] = False

    # Add bot response to chat history
    bot_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(),
        "show_feedback": show_feedback
    }
    session_state["chat_history"].append(bot_message)

    # Don't send suggestions with response anymore since we're doing real-time
    # Mobile-optimized response with additional metadata
    return {
        "response": response,
        "show_feedback": show_feedback,
        "response_type": "text",  # Can be text, card, list, etc.
        "quick_actions": get_mobile_quick_actions(response),  # Quick action buttons for mobile
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback(request: FeedbackRequest):
    """
    Process user feedback on AI responses.

    Args:
        request (FeedbackRequest): The feedback request containing feedback type and message index.

    Returns:
        FeedbackResponse: The response to the user feedback.
    """
    if request.feedback == "no":
        # Don't create ticket anymore, just provide contact info
        response = """Contact Us:
support@novarsistech.com"""
        session_state["resolved_count"] -= 1
    else:
        if session_state.get("platform") == "mobile":
            response = "Great! Happy to help! 😊"
        else:
            response = "Great! I'm glad I could help. Feel free to ask if you have any more questions about Novarsis! 🚀"
        session_state["resolved_count"] += 1

    bot_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(),
        "show_feedback": True  # Changed to True
    }
    session_state["chat_history"].append(bot_message)

    return {"response": response}


@app.post("/api/upload", response_model=UploadResponse, tags=["File Upload"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an image file for SEO analysis.

    Args:
        file (UploadFile): The image file to upload.

    Returns:
        UploadResponse: Information about the uploaded file.

    Raises:
        HTTPException: If no file is uploaded or if the file type is invalid.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only image files (JPG, JPEG, PNG, GIF, WEBP) are allowed")

    # Read file and convert to base64
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')

    # Log that an image was uploaded
    logger.info(f"Image uploaded for analysis: {file.filename}")

    # Return with metadata for SEO error detection
    return {
        "image_data": base64_image,
        "filename": file.filename,
        "content_type": file.content_type,
        "instructions": "Please attach this image and describe the SEO error you're seeing for best results."
    }


@app.get("/api/chat-history", response_model=ChatHistoryResponse, tags=["Chat"])
async def get_chat_history():
    """
    Retrieve the chat history for the current session.

    Returns:
        ChatHistoryResponse: The chat history containing all messages.
    """
    return {"chat_history": session_state["chat_history"]}


@app.post("/api/mobile/chat", response_model=MobileChatResponse, tags=["Mobile"])
async def mobile_chat(request: ChatRequest):
    """Mobile-specific chat endpoint with optimized responses"""
    request.platform = "mobile"  # Force mobile platform

    # Process the chat request
    response = await chat(request)

    # Format response for mobile
    mobile_response = {
        "status": "success",
        "data": {
            "message": response["response"],
            "message_id": f"msg_{datetime.now().timestamp()}",
            "timestamp": response["timestamp"],
            "type": response["response_type"],
            "quick_actions": response.get("quick_actions", []),
            "suggestions": get_context_suggestions(request.message)[:3],  # Max 3 for mobile
            "metadata": {
                "show_feedback": response["show_feedback"],
                "requires_action": bool(response.get("quick_actions")),
                "session_id": session_state.get("session_id", "default")
            }
        }
    }

    return mobile_response


@app.get("/api/mobile/suggestions", response_model=MobileSuggestionsResponse, tags=["Mobile"])
async def get_mobile_suggestions():
    """
    Get mobile-optimized quick action suggestions.

    Returns:
        MobileSuggestionsResponse: A list of suggested actions for mobile users.
    """
    return {
        "status": "success",
        "data": {
            "suggestions": [
                {"text": "🔍 Check SEO Score", "id": "seo_check"},
                {"text": "💳 View Plans", "id": "view_plans"},
                {"text": "📞 Contact Support", "id": "contact"}
            ]
        }
    }


@app.post("/api/mobile/quick-action", response_model=QuickActionResponse, tags=["Mobile"])
async def handle_quick_action(request: dict):
    """
    Handle quick action button clicks from mobile interface.

    Args:
        request (dict): The action request containing the action type.

    Returns:
        QuickActionResponse: The result of the quick action.
    """
    action = request.get("action", "")

    if action == "call_support":
        return {
            "status": "success",
            "data": {
                "action": "call",
                "number": WHATSAPP_NUMBER
            }
        }
    elif action == "view_report":
        return {
            "status": "success",
            "data": {
                "action": "navigate",
                "screen": "reports"
            }
        }
    elif action == "upgrade_plan":
        return {
            "status": "success",
            "data": {
                "action": "navigate",
                "screen": "pricing",
                "params": {"highlight": "pro"}
            }
        }
    else:
        return {
            "status": "success",
            "data": {
                "action": "continue_chat"
            }
        }


@app.get("/api/suggestions", response_model=SuggestionsResponse, tags=["Suggestions"])
async def get_suggestions():
    """
    Get initial suggestions when the chat loads.

    Returns:
        SuggestionsResponse: A list of initial suggestions.
    """
    # Return empty suggestions initially - don't show anything until user types
    return {"suggestions": []}


@app.post("/api/typing-suggestions", response_model=TypingSuggestionsResponse, tags=["Suggestions"])
async def get_typing_suggestions(request: dict):
    """
    Get real-time suggestions based on what user is typing.

    Args:
        request (dict): The request containing the current input text.

    Returns:
        TypingSuggestionsResponse: Context-aware suggestions based on input.
    """
    user_input = request.get("input", "")

    # Only show suggestions if user has typed at least 3 characters
    # This prevents suggestions from appearing immediately
    if len(user_input.strip()) < 3:
        return {"suggestions": []}

    suggestions = get_context_suggestions(user_input)
    return {"suggestions": suggestions}


# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create index.html template
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Novarsis Support Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Inter', sans-serif !important;
        }

        body {
            background: #f0f2f5;
            margin: 0;
            padding: 0;
        }

        .main-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }

        .header-container {
            background: white;
            border-radius: 16px;
            padding: 16px 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-section {
            display: flex;
            align-items: center;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .contact-btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .contact-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .contact-popup {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            z-index: 1000;
            display: none;
            text-align: center;
        }

        .contact-popup.show {
            display: block;
            animation: popIn 0.3s ease;
        }

        @keyframes popIn {
            from {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
            to {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
        }

        .contact-popup h3 {
            margin-top: 0;
            color: #333;
            font-size: 20px;
        }

        .contact-email {
            font-size: 18px;
            color: #667eea;
            font-weight: 600;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            user-select: all;
            cursor: pointer;
        }

        .copy-btn {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px 5px;
        }

        .copy-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .close-popup-btn {
            padding: 10px 20px;
            background: #f1f3f5;
            color: #333;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px 5px;
        }

        .close-popup-btn:hover {
            background: #e1e4e8;
        }

        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 999;
            display: none;
        }

        .overlay.show {
            display: block;
        }

        .logo {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-right: 10px;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            background: #e8f5e9;
            border-radius: 20px;
            font-size: 13px;
            color: #2e7d32;
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #4caf50;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .chat-container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            height: 70vh;
            min-height: 500px;
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 20px;
            position: relative;
        }

        .message-wrapper {
            display: flex;
            margin-bottom: 20px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .user-message-wrapper {
            justify-content: flex-end;
        }

        .bot-message-wrapper {
            justify-content: flex-start;
        }

        .message-content {
            max-width: 70%;
            min-width: min-content;
            width: fit-content;
            padding: 16px 20px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.6;
            position: relative;
            word-wrap: break-word;
            white-space: pre-wrap;
        }

        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        .bot-message {
            background: #f1f3f5;
            color: #2d3436;
            border-bottom-left-radius: 5px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin: 0 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 16px;
            flex-shrink: 0;
        }

        .user-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .bot-avatar {
            background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
            color: white;
        }

        .timestamp {
            font-size: 11px;
            color: rgba(0,0,0,0.5);
            margin-top: 8px;
            font-weight: 400;
        }

        .user-timestamp {
            color: rgba(255,255,255,0.8);
            text-align: right;
        }

        .typing-indicator {
            display: flex;
            align-items: center;
            padding: 15px;
            background: #f1f3f5;
            border-radius: 18px;
            width: fit-content;
            margin-left: 64px;
            margin-bottom: 20px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: #95a5a6;
            border-radius: 50%;
            margin: 0 3px;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }

        .input-container {
            background: white;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            position: sticky;
            bottom: 20px;
        }

        .suggestions-container {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            max-height: 80px;
            overflow-y: auto;
            padding: 4px 0;
            transition: opacity 0.15s ease;
            min-height: 32px;
        }

        .suggestion-pill {
            padding: 8px 14px;
            background: #f0f2f5;
            border: 1px solid #e1e4e8;
            border-radius: 20px;
            font-size: 13px;
            color: #24292e;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
            flex-shrink: 0;
            font-weight: 500;
            animation: slideInFade 0.3s ease-out forwards;
            opacity: 0;
        }

        @keyframes slideInFade {
            from {
                opacity: 0;
                transform: translateY(-5px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .suggestion-pill:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(102, 126, 234, 0.2);
        }

        .suggestion-pill:active {
            transform: translateY(0);
        }

        .suggestions-container::-webkit-scrollbar {
            height: 4px;
        }

        .suggestions-container::-webkit-scrollbar-track {
            background: transparent;
        }

        .suggestions-container::-webkit-scrollbar-thumb {
            background: #d0d0d0;
            border-radius: 2px;
        }

        .message-form {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .message-input {
            flex: 1;
            border-radius: 24px;
            border: 1px solid #e0e0e0;
            padding: 14px 20px;
            font-size: 15px;
            background: #f8f9fa;
            color: #333333;
            outline: none;
        }

        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }

        .send-btn {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .attachment-btn {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #54656f;
            padding: 0;
        }

        .attachment-btn:hover {
            background-color: #f1f3f5;
            border-color: #667eea;
            transform: scale(1.05);
        }

        .attachment-btn.success {
            background-color: #e8f5e9;
            color: #4caf50;
            border-color: #4caf50;
            pointer-events: none;
        }

        .attachment-btn.success svg path {
            fill: #4caf50;
        }

        .feedback-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            margin-left: 64px;
        }

        .feedback-btn {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            border: 1px solid #e0e0e0;
            background: white;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .feedback-btn:hover {
            background: #f8f9fa;
            border-color: #667eea;
        }

        .file-input {
            display: none;
        }

        /* Initial message styling - Ultra Compact */
        .initial-message .message-content {
            padding: 8px 12px !important;
            line-height: 1.2 !important;
            max-width: max-content !important;
            min-width: unset !important;
            width: max-content !important;
            display: inline-block !important;
            font-size: 14px !important;
        }

        .initial-message.bot-message-wrapper {
            display: flex;
            align-items: flex-start;
            margin-bottom: 15px;
        }

        .initial-message .avatar {
            width: 32px;
            height: 32px;
            font-size: 13px;
            margin-right: 8px;
            flex-shrink: 0;
        }

        .initial-message .timestamp {
            font-size: 10px;
            color: rgba(0,0,0,0.4);
            margin-top: 3px;
            display: block;
        }

        /* Force initial bot message to be compact */
        .initial-message .bot-message {
            max-width: max-content !important;
            width: max-content !important;
            display: inline-block !important;
            white-space: nowrap !important;
        }

        /* Allow timestamp to wrap normally */
        .initial-message .bot-message .timestamp {
            white-space: normal !important;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-container {
                padding: 10px;
            }

            .chat-container {
                height: 65vh;
                border-radius: 12px;
                padding: 15px;
            }

            .message-content {
                max-width: 80%;
                font-size: 14px;
            }

            .input-container {
                padding: 12px;
                border-radius: 12px;
            }

            .header-container {
                padding: 12px 16px;
                border-radius: 12px;
            }

            .avatar {
                width: 36px;
                height: 36px;
                font-size: 14px;
            }

            .typing-indicator {
                margin-left: 52px;
            }
        }
    </style>
</head>
<body>
    <div class="overlay" id="overlay"></div>

    <div class="contact-popup" id="contactPopup">
        <h3>📧 Contact Support</h3>
        <div class="contact-email" id="contactEmail">support@novarsistech.com</div>
        <button class="copy-btn" onclick="copyEmail()">📋 Copy Email</button>
        <button class="close-popup-btn" onclick="closeContactPopup()">Close</button>
    </div>

    <div class="main-container">
        <div class="header-container">
            <div class="logo-section">
                <span class="logo">🚀 NOVARSIS</span>
                <span style="color: #95a5a6; font-size: 14px; margin-left: 10px;">AI Support Center</span>
            </div>
            <div class="header-right">
                <button class="contact-btn" onclick="showContactPopup()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M22 6l-10 7L2 6" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Contact Us
                </button>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>Nova is Online</span>
                </div>
            </div>
        </div>

        <div class="chat-container" id="chat-container">
            <!-- Initial greeting message -->
            <div class="message-wrapper bot-message-wrapper initial-message">
                <div class="avatar bot-avatar">N</div>
                <div class="message-content bot-message">
                    Hi, I am Nova, How may I assist you today?
                    <div class="timestamp bot-timestamp">Now</div>
                </div>
            </div>
        </div>

        <div class="input-container">
            <div class="suggestions-container" id="suggestions-container">
                <!-- Initial quick response suggestions will be dynamically added here -->
            </div>

            <form class="message-form" id="message-form">
                <input type="file" id="file-input" class="file-input" accept="image/jpeg,image/jpg,image/png,image/gif,image/webp">
                <button type="button" class="attachment-btn" id="attachment-btn">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1 -1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" fill="currentColor"/>
                    </svg>
                </button>
                <input type="text" class="message-input" id="message-input" placeholder="Type your message...">
                <button type="submit" class="send-btn">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="white"/>
                    </svg>
                </button>
            </form>
        </div>
    </div>

    <script>
        // Contact popup functions
        function showContactPopup() {
            document.getElementById('contactPopup').classList.add('show');
            document.getElementById('overlay').classList.add('show');
        }

        function closeContactPopup() {
            document.getElementById('contactPopup').classList.remove('show');
            document.getElementById('overlay').classList.remove('show');
        }

        function copyEmail() {
            const email = 'support@novarsistech.com';
            navigator.clipboard.writeText(email).then(() => {
                const copyBtn = event.target;
                const originalText = copyBtn.textContent;
                copyBtn.textContent = '✓ Copied!';
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                }, 2000);
            });
        }

        // Close popup when clicking overlay
        document.getElementById('overlay').addEventListener('click', closeContactPopup);

        // Format time function
        function formatTime(timestamp) {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        }

        // Current time for welcome message
        document.addEventListener('DOMContentLoaded', function() {
            // Set current time for initial greeting
            const initialTimestamp = document.querySelector('.initial-message .timestamp');
            if (initialTimestamp) {
                initialTimestamp.textContent = formatTime(new Date());
            }

            // Load initial suggestions
            loadInitialSuggestions();
        });

        // Chat container
        const chatContainer = document.getElementById('chat-container');

        // Message input
        const messageForm = document.getElementById('message-form');
        const messageInput = document.getElementById('message-input');
        const attachmentBtn = document.getElementById('attachment-btn');
        const fileInput = document.getElementById('file-input');

        // Suggestions container
        const suggestionsContainer = document.getElementById('suggestions-container');

        // File handling
        let uploadedImageData = null;
        let uploadedFileName = '';

        attachmentBtn.addEventListener('click', function() {
            fileInput.click();
        });

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    uploadedImageData = event.target.result.split(',')[1]; // Get base64 data
                    uploadedFileName = file.name;
                    attachmentBtn.classList.add('success');
                    // Change icon to checkmark
                    attachmentBtn.innerHTML = `
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill="currentColor"/>
                        </svg>
                    `;
                };
                reader.readAsDataURL(file);
            }
        });

        // Add message to chat
        function addMessage(role, content, showFeedback = true) {
            const messageWrapper = document.createElement('div');
            messageWrapper.className = `message-wrapper ${role}-message-wrapper`;

            const avatar = document.createElement('div');
            avatar.className = `avatar ${role}-avatar`;
            avatar.textContent = role === 'user' ? '@' : 'N';

            const messageContent = document.createElement('div');
            messageContent.className = `message-content ${role}-message`;
            // Set textContent to preserve formatting
            messageContent.textContent = content;

            const timestamp = document.createElement('div');
            timestamp.className = `timestamp ${role}-timestamp`;
            timestamp.textContent = formatTime(new Date());

            messageContent.appendChild(timestamp);

            if (role === 'user') {
                messageWrapper.appendChild(messageContent);
                messageWrapper.appendChild(avatar);
            } else {
                messageWrapper.appendChild(avatar);
                messageWrapper.appendChild(messageContent);
                // Feedback buttons removed: assistant messages now only show avatar and content.
            }

            chatContainer.appendChild(messageWrapper);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Show typing indicator
        function showTypingIndicator() {
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'typing-indicator';
            typingIndicator.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            `;
            chatContainer.appendChild(typingIndicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return typingIndicator;
        }

        // Update suggestions with smooth animation
        function updateSuggestions(suggestions) {
            const container = document.getElementById('suggestions-container');

            // Smooth transition
            container.style.opacity = '0';

            setTimeout(() => {
                container.innerHTML = '';

                if (suggestions && suggestions.length > 0) {
                    suggestions.forEach((suggestion, index) => {
                        const pill = document.createElement('div');
                        pill.className = 'suggestion-pill';
                        pill.textContent = suggestion;
                        pill.style.animationDelay = `${index * 50}ms`;
                        pill.onclick = () => {
                            messageInput.value = suggestion;
                            messageForm.dispatchEvent(new Event('submit'));
                        };
                        container.appendChild(pill);
                    });
                }

                container.style.opacity = '1';
            }, 150);
        }

        // Load initial suggestions
        function loadInitialSuggestions() {
            // Load initial quick response suggestions
            const initialSuggestions = [
                "How do I analyze my website SEO?",
                "Check my subscription status",
                "I'm getting an error message",
                "Generate SEO report",
                "Compare pricing plans",
            ];

            updateSuggestions(initialSuggestions);
        }

        /**
         * DEBOUNCING IMPLEMENTATION
         * - Suggestions API call fires only after user stops typing for 500ms
         * - Every keystroke clears the previous timer and sets a new one
         * - Example: User types "car" quickly:
         *   - 'c' typed → timer starts
         *   - 'a' typed → timer resets
         *   - 'r' typed → timer resets
         *   - User stops → after 500ms, ONE API call is made with "car"
         * - This prevents multiple API calls and improves performance
         */

        // Typing suggestions with debouncing - 500ms after user stops typing
        let typingTimer;
        const DEBOUNCE_DELAY = 500; // 500ms debounce delay

        async function fetchTypingSuggestions(input) {
            // Require at least 3 characters before showing suggestions
            if (input.trim().length < 3) {
                // Clear suggestions if input is too short
                updateSuggestions([]);
                return;
            }

            try {
                const response = await fetch('/api/typing-suggestions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input: input })
                });

                const data = await response.json();
                updateSuggestions(data.suggestions);
            } catch (error) {
                console.error('Error fetching suggestions:', error);
            }
        }

        // Handle input changes with debouncing
        messageInput.addEventListener('input', function(e) {
            const inputValue = e.target.value;

            // Clear existing timer (debouncing)
            clearTimeout(typingTimer);

            // Clear suggestions immediately while typing
            updateSuggestions([]);

            // Set new timer - execute after user stops typing for 500ms
            typingTimer = setTimeout(() => {
                // Only fetch suggestions if input has at least 3 characters
                if (inputValue.trim().length >= 3) {
                    fetchTypingSuggestions(inputValue);
                } else {
                    // If input is cleared or too short, show initial suggestions again
                    if (inputValue.trim() === '') {
                        loadInitialSuggestions();
                    } else {
                        // Keep suggestions empty for short input
                        updateSuggestions([]);
                    }
                }
            }, DEBOUNCE_DELAY);
        });

        // Handle focus - show initial suggestions
        messageInput.addEventListener('focus', function(e) {
            // If input is empty, show initial suggestions
            if (messageInput.value.trim() === '') {
                loadInitialSuggestions();
            }
        });

        // Handle blur - if input is empty, show initial suggestions
        messageInput.addEventListener('blur', function(e) {
            // Small delay to allow click events on suggestions to fire
            setTimeout(() => {
                if (messageInput.value.trim() === '') {
                    loadInitialSuggestions();
                }
            }, 200);
        });

        // Send message
        async function sendMessage(message, imageData = null) {
            // Handle special commands - ticket system removed
            // No special commands currently implemented

            if (message.toLowerCase() === 'connect with an expert') {
                // Clear suggestions
                updateSuggestions([]);

                // Call the connect expert API
                try {
                    const response = await fetch('/api/connect-expert', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    addMessage('assistant', data.response, true);
                } catch (error) {
                    console.error('Error connecting with expert:', error);
                    addMessage('assistant', 'Sorry, I encountered an error connecting you with an expert.', true);
                }

                // Load initial suggestions after a delay
                setTimeout(() => {
                    loadInitialSuggestions();
                }, 500);
                return;
            }

            // Normal message handling
            // Add user message
            addMessage('user', message);

            // Clear suggestions after sending
            updateSuggestions([]);

            // Show typing indicator
            const typingIndicator = showTypingIndicator();

            try {
                // Send to API
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        image_data: imageData
                    })
                });

                const data = await response.json();

                // Remove typing indicator
                typingIndicator.remove();

                // Add bot response
                addMessage('assistant', data.response, data.show_feedback);

                // Load initial suggestions after response
                setTimeout(() => {
                    loadInitialSuggestions();
                }, 500);

                // Reset attachment
                if (uploadedImageData) {
                    attachmentBtn.classList.remove('success');
                    attachmentBtn.innerHTML = `
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1 -1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" fill="currentColor"/>
                        </svg>
                    `;
                    uploadedImageData = null;
                    uploadedFileName = '';
                    fileInput.value = '';
                }

            } catch (error) {
                console.error('Error sending message:', error);
                typingIndicator.remove();
                addMessage('assistant', 'Sorry, I encountered an error. Please try again.', true);
                // Load initial suggestions on error
                setTimeout(() => {
                    loadInitialSuggestions();
                }, 500);
            }
        }

        // Send feedback
        async function sendFeedback(feedback) {
            const messageIndex = document.querySelectorAll('.message-wrapper').length - 1;

            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        feedback: feedback,
                        message_index: messageIndex
                    })
                });

                const data = await response.json();
                addMessage('assistant', data.response, true);

            } catch (error) {
                console.error('Error sending feedback:', error);
            }
        }

        // Handle form submission
        messageForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const message = messageInput.value.trim();
            if (message) {
                await sendMessage(message, uploadedImageData);
                messageInput.value = '';
            }
        });

        // Handle Enter key in message input
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                messageForm.dispatchEvent(new Event('submit'));
            }
        });
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
