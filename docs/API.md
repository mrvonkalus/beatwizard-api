# 🎵 BeatWizard API Documentation

## Base URL
```
Production: https://beatwizard-api.onrender.com
Development: http://localhost:8080
```

## Authentication
Currently using public endpoints. Supabase authentication integration planned.

## Core Endpoints

### 🏥 Health Check
```http
GET /
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "BeatWizard Audio Analysis API",
  "version": "1.0.0-minimal",
  "timestamp": 1755673733.5787997
}
```

### 📊 Service Status
```http
GET /api/status
```
**Response:**
```json
{
  "current_phase": "phase_1_m4a_enhanced",
  "audio_processing": "full_enabled",
  "nlu_conversation": "enabled",
  "full_analyzer": {"enabled": true},
  "limits": {"max_upload_mb": 25}
}
```

### 📝 Service Information
```http
GET /api/info
```
**Response:**
```json
{
  "api_name": "BeatWizard Audio Analysis",
  "version": "1.0.0",
  "available_endpoints": [...],
  "audio_features": {
    "status": "production_ready",
    "features": [
      "Tempo detection",
      "Key detection", 
      "7-band frequency analysis",
      "LUFS measurement",
      "Dynamic range analysis",
      "Stereo field analysis",
      "Intelligent AI feedback"
    ]
  }
}
```

## 🎵 Audio Analysis

### Full Analysis
```http
POST /api/analyze
Content-Type: multipart/form-data
```

**Request:**
```javascript
const formData = new FormData();
formData.append('file', audioFile);

fetch('/api/analyze', {
  method: 'POST',
  body: formData
});
```

**Supported Formats:**
- ✅ WAV (best compatibility)
- ✅ FLAC (lossless quality)  
- ✅ MP3 (wide support)
- ✅ OGG (open format)
- ⚡ M4A (with fallback conversion)

**File Limits:**
- Max size: 25MB
- Max duration: 10 minutes
- Sample rates: 22kHz - 192kHz

**Response:**
```json
{
  "ok": true,
  "analysis": {
    "tempo_analysis": {
      "bpm": 128.5,
      "confidence": 0.95,
      "time_signature": "4/4"
    },
    "key_analysis": {
      "key": "A",
      "mode": "minor", 
      "confidence": 0.87
    },
    "loudness_analysis": {
      "integrated_loudness": -14.2,
      "peak_level": -0.1,
      "dynamic_range": 12.3,
      "quality_rating": "excellent"
    },
    "frequency_analysis": {
      "frequency_bands": {
        "sub_bass": {"energy": 145.2, "range": "20-60Hz"},
        "bass": {"energy": 289.7, "range": "60-250Hz"},
        "low_mid": {"energy": 156.3, "range": "250Hz-1kHz"},
        "mid": {"energy": 198.4, "range": "1-4kHz"},
        "high_mid": {"energy": 89.6, "range": "4-8kHz"},
        "presence": {"energy": 45.2, "range": "8-16kHz"},
        "brilliance": {"energy": 23.1, "range": "16kHz+"}
      },
      "overall_assessment": {
        "quality_rating": "professional",
        "balance_score": 8.5
      }
    },
    "stereo_analysis": {
      "stereo_width": 0.75,
      "correlation": 0.82,
      "phase_issues": false
    },
    "metadata": {
      "duration": 180.5,
      "sample_rate": 44100,
      "channels": 2,
      "file_size": 15728640
    }
  },
  "processing_time": 2.34,
  "analysis_id": "uuid-string"
}
```

### Lite Analysis
```http
POST /api/analyze-lite
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "ok": true,
  "analysis": {
    "duration_sec": 180.5,
    "peak": 0.95,
    "rms": 0.15,
    "crest_factor_db": 15.2,
    "metadata": {
      "samplerate": 44100,
      "channels": 2,
      "format": "WAV"
    }
  }
}
```

## 🧙‍♂️ AI Conversation System

### Start Conversation
```http
POST /api/conversation/start
Content-Type: application/json
```

**Request:**
```json
{
  "skill_level": "intermediate",
  "genre": "electronic",
  "analysis_results": {...}
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "welcome_message": "🧙‍♂️ *adjusts wizard hat* Welcome, skilled practitioner!...",
  "initial_suggestions": [
    "How can I improve my mix?",
    "What's the genre of my track?",
    "How are my vocals sitting?"
  ]
}
```

### Send Message
```http
POST /api/conversation/message
Content-Type: application/json
```

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "How does my mix sound?",
  "context": {
    "skill_level": "intermediate",
    "analysis_results": {...}
  }
}
```

**Response:**
```json
{
  "response": {
    "content": "🧙‍♂️ *adjusts wizard hat with knowing confidence* Ah, a skilled practitioner! Let me unveil the deeper mysteries within your sonic realm...",
    "metadata": {
      "intent": "mix_analysis",
      "confidence": 0.92,
      "elements_discussed": ["loudness", "frequency_balance", "dynamics"],
      "response_style": "intermediate_detailed"
    },
    "components": {
      "actionable_steps": [
        "Boost your track to -14 LUFS for streaming",
        "Check 200-400Hz for muddiness",
        "Add high-shelf at 10kHz for air"
      ],
      "supporting_details": [...],
      "follow_up_questions": [...],
      "encouragement": "Your mix shows great potential!"
    }
  },
  "session_updated": true
}
```

## ❌ Error Responses

### File Format Error
```json
{
  "error": "M4A decoder error",
  "detail": "M4A format requires additional codecs...",
  "suggestion": "For best results, please convert to WAV or FLAC format...",
  "supported_formats": ["WAV", "FLAC", "MP3", "OGG"],
  "status": "ffmpeg_dependency_issue"
}
```

### File Size Error
```json
{
  "error": "File too large",
  "detail": "File exceeds 25MB limit",
  "max_size_mb": 25,
  "received_size_mb": 47.3
}
```

### Processing Error
```json
{
  "error": "Analysis failed",
  "detail": "Could not extract audio features",
  "suggestion": "Try a different audio file or format",
  "support_email": "support@beatwizard.app"
}
```

## 📊 Rate Limits
- 100 requests per minute per IP
- 10 analysis requests per minute per IP
- 1GB total upload per hour per IP

## 🔧 CORS Policy
```
Allowed Origins: *
Allowed Methods: GET, POST, OPTIONS
Allowed Headers: Content-Type, Authorization
```

## 📈 Response Times
- Health checks: < 100ms
- Lite analysis: < 2 seconds
- Full analysis: 2-10 seconds (depending on file size)
- AI responses: 1-3 seconds

## 🛠️ Development Testing

### cURL Examples
```bash
# Health check
curl https://beatwizard-api.onrender.com/health

# File analysis
curl -X POST \
  -F "file=@track.wav" \
  https://beatwizard-api.onrender.com/api/analyze

# AI conversation
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"skill_level":"beginner","message":"What is LUFS?"}' \
  https://beatwizard-api.onrender.com/api/conversation/message
```

### JavaScript Examples
```javascript
// File upload with progress
const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/analyze', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};

// AI conversation
const chatWithWizard = async (message, sessionId) => {
  const response = await fetch('/api/conversation/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
      context: { skill_level: 'intermediate' }
    })
  });
  
  return await response.json();
};
```

---

**Last Updated:** August 2024  
**API Version:** 1.0.0  
**Status:** Production Ready 🚀
