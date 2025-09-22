from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import time
import random
import google.generativeai as genai
from datetime import datetime, timedelta
import base64
import io
from PIL import Image
import hashlib
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Novarsis Support Center API",
    description="AI Support Assistant API for Novarsis SEO Tool",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== CONFIGURATION ======================

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
try:
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    logger.info("✅ Gemini model initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini model: {str(e)}")
    model = None

# Support Configuration
WHATSAPP_NUMBER = "+91-9999999999"
SUPPORT_EMAIL = "support@novarsis.tech"

# ====================== PYDANTIC MODELS ======================

class ChatRequest(BaseModel):
    message: str
    image_data: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    show_feedback: bool = True
    suggestions: List[str] = []
    session_id: str
    ticket_id: Optional[str] = None

class TicketStatusRequest(BaseModel):
    ticket_id: str

class TicketStatusResponse(BaseModel):
    ticket_id: str
    status: str
    priority: str
    created_at: str
    query: str
    response_time: str

class FeedbackRequest(BaseModel):
    feedback: str  # "yes" or "no"
    message_index: int
    session_id: str

class ConnectExpertRequest(BaseModel):
    query: str
    session_id: str

class SuggestionsRequest(BaseModel):
    input: str
    session_id: Optional[str] = None

class UploadResponse(BaseModel):
    image_data: str
    filename: str
    size: int

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    gemini_status: str
    version: str

# ====================== SESSION MANAGEMENT ======================

# In-memory session storage (use Redis in production)
sessions = {}

def get_or_create_session(session_id: Optional[str] = None) -> Dict:
    """Get existing session or create new one"""
    if not session_id:
        session_id = f"session_{random.randint(100000, 999999)}_{int(time.time())}"
    
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "chat_history": [],
            "support_tickets": {},
            "current_query": {},
            "user_name": "User",
            "session_start": datetime.now(),
            "last_activity": datetime.now(),
            "resolved_count": 0,
            "checking_ticket_status": False,
            "last_user_query": ""
        }
    else:
        sessions[session_id]["last_activity"] = datetime.now()
    
    return sessions[session_id]

def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    current_time = datetime.now()
    expired = []
    for sid, session in sessions.items():
        if (current_time - session["last_activity"]).days > 1:
            expired.append(sid)
    
    for sid in expired:
        del sessions[sid]
    
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")

# ====================== AI RESPONSE LOGIC ======================

SYSTEM_PROMPT = """You are Nova, the official AI support assistant for Novarsis AIO SEO Tool.

Be conversational, friendly, and helpful. Keep responses concise (2-4 lines for simple queries).
Answer questions about SEO analysis, reports, accounts, billing, and technical issues.

For unrelated queries (cooking, movies, etc.), politely say:
"Sorry, I only help with Novarsis SEO Tool. Please let me know if you have any SEO tool related questions?"
"""

def get_ai_response(user_input: str, session: Dict, image_data: Optional[str] = None) -> str:
    """Generate AI response using Gemini"""
    if not model:
        return "I'm having trouble connecting to my AI service. Please try again or contact support."
    
    try:
        # Build context from chat history
        context = ""
        if session["chat_history"]:
            recent = session["chat_history"][-5:]
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"
        
        prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n{context}\n\nUser: {user_input}"
        
        if image_data:
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I encountered an issue processing your request. Please try again."

def get_suggestions(user_input: str = "") -> List[str]:
    """Get context-based suggestions"""
    if not user_input:
        return [
            "How do I analyze my website SEO?",
            "Check my subscription status",
            "I'm getting an error message",
            "Generate SEO report",
            "Connect with an Expert"
        ]
    
    input_lower = user_input.lower()
    
    if "error" in input_lower or "problem" in input_lower:
        return [
            "Website not loading",
            "Analysis stuck at 0%",
            "Can't access reports",
            "Connect with an Expert"
        ]
    elif "seo" in input_lower:
        return [
            "How to improve SEO score?",
            "Check page load speed",
            "Analyze competitor websites",
            "Mobile optimization tips"
        ]
    elif "account" in input_lower or "billing" in input_lower:
        return [
            "Upgrade my plan",
            "View billing history",
            "Cancel subscription",
            "Update payment method"
        ]
    else:
        return []

def create_support_ticket(query: str, session: Dict) -> str:
    """Create a support ticket"""
    ticket_id = f"NVS{random.randint(10000, 99999)}"
    ticket = {
        "id": ticket_id,
        "query": query,
        "status": "In Progress",
        "priority": "High" if "urgent" in query.lower() else "Normal",
        "created_at": datetime.now(),
        "response_time": "Within 15 minutes"
    }
    session["support_tickets"][ticket_id] = ticket
    return ticket_id

# ====================== API ENDPOINTS ======================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns API documentation link"""
    return """
    <html>
        <body>
            <h1>Novarsis Support Center API</h1>
            <p>API Documentation: <a href="/api/docs">/api/docs</a></p>
            <p>ReDoc: <a href="/api/redoc">/api/redoc</a></p>
        </body>
    </html>
    """

@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        gemini_status="connected" if model else "disconnected",
        version="1.0.0"
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    # Get or create session
    session = get_or_create_session(request.session_id)
    
    # Add user message to history
    session["chat_history"].append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now()
    })
    
    # Store last query
    session["last_user_query"] = request.message
    
    # Generate AI response
    response_text = get_ai_response(request.message, session, request.image_data)
    
    # Add bot response to history
    session["chat_history"].append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now()
    })
    
    # Get suggestions
    suggestions = get_suggestions(request.message)
    
    return ChatResponse(
        response=response_text,
        show_feedback=True,
        suggestions=suggestions,
        session_id=session["id"]
    )

@app.post("/api/check-ticket-status", response_model=TicketStatusResponse)
async def check_ticket_status(request: TicketStatusRequest):
    """Check support ticket status"""
    # Search across all sessions for ticket
    for session in sessions.values():
        if request.ticket_id in session["support_tickets"]:
            ticket = session["support_tickets"][request.ticket_id]
            return TicketStatusResponse(
                ticket_id=request.ticket_id,
                status=ticket["status"],
                priority=ticket["priority"],
                created_at=ticket["created_at"].isoformat(),
                query=ticket["query"],
                response_time=ticket["response_time"]
            )
    
    raise HTTPException(status_code=404, detail=f"Ticket {request.ticket_id} not found")

@app.post("/api/connect-expert")
async def connect_expert(request: ConnectExpertRequest):
    """Connect with human expert"""
    session = get_or_create_session(request.session_id)
    
    # Create support ticket
    ticket_id = create_support_ticket(request.query, session)
    
    response = f"""I've created a priority support ticket for you:

🎫 Ticket ID: {ticket_id}
📱 Status: Escalated to Human Support
⏱️ Response Time: Within 15 minutes

Our expert team will reach out via:
• In-app chat
• Email: {SUPPORT_EMAIL}
• WhatsApp: {WHATSAPP_NUMBER}"""
    
    return {
        "response": response,
        "ticket_id": ticket_id,
        "session_id": session["id"]
    }

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on response"""
    session = get_or_create_session(request.session_id)
    
    if request.feedback == "no":
        # Create support ticket for unresolved issue
        last_query = session.get("last_user_query", "User query")
        ticket_id = create_support_ticket(last_query, session)
        
        response = f"""I understand this didn't fully resolve your issue. I've created a priority support ticket:

🎫 Ticket ID: {ticket_id}
Our team will contact you within 15 minutes."""
        
        session["resolved_count"] -= 1
    else:
        response = "Great! I'm glad I could help. Feel free to ask if you have any more questions!"
        session["resolved_count"] += 1
    
    return {
        "response": response,
        "session_id": session["id"]
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload image file"""
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG files are allowed")
    
    # Check file size (max 5MB)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")
    
    # Convert to base64
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    return UploadResponse(
        image_data=base64_image,
        filename=file.filename,
        size=len(contents)
    )

@app.get("/api/suggestions")
async def get_initial_suggestions():
    """Get initial chat suggestions"""
    return {"suggestions": get_suggestions()}

@app.post("/api/typing-suggestions")
async def get_typing_suggestions(request: SuggestionsRequest):
    """Get real-time typing suggestions"""
    suggestions = get_suggestions(request.input)
    return {"suggestions": suggestions}

@app.get("/api/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    session = get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "chat_history": session["chat_history"],
        "session_start": session["session_start"].isoformat()
    }

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    cleanup_old_sessions()
    
    total_sessions = len(sessions)
    total_messages = sum(len(s["chat_history"]) for s in sessions.values())
    total_tickets = sum(len(s["support_tickets"]) for s in sessions.values())
    
    return {
        "active_sessions": total_sessions,
        "total_messages": total_messages,
        "total_tickets": total_tickets,
        "timestamp": datetime.now().isoformat()
    }

# ====================== ERROR HANDLERS ======================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# ====================== STARTUP/SHUTDOWN ======================

@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    logger.info("🚀 Novarsis Support Center API started")
    logger.info(f"📚 API Docs available at: /api/docs")
    logger.info(f"🔍 ReDoc available at: /api/redoc")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on app shutdown"""
    logger.info("👋 Novarsis Support Center API shutting down")

# ====================== MAIN ======================

if __name__ == "__main__":
    uvicorn.run(
        "api_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
