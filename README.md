# 🧙‍♂️ BeatWizard - AI-Powered Music Production Assistant

[![Deploy Status](https://img.shields.io/badge/deploy-live-brightgreen)](https://beatwizard-api.onrender.com)
[![Frontend](https://img.shields.io/badge/frontend-react-blue)](http://localhost:3000)
[![AI Powered](https://img.shields.io/badge/AI-powered-purple)](https://openai.com)

> Transform your music production with magical AI assistance. Upload tracks, get professional analysis, and chat with an intelligent wizard who understands your music.

## ✨ Features

- 🎵 **Professional Audio Analysis** - 7-band EQ, LUFS, dynamic range, stereo analysis
- 🧙‍♂️ **AI Wizard Chat** - Intelligent conversation with skill-level adaptation
- 📱 **Mobile Optimized** - Perfect experience on all devices
- 🎪 **Dynamic Responses** - Never repetitive, always engaging
- ⚡ **Multiple Formats** - WAV, FLAC, MP3, OGG, M4A support
- 🎯 **Production Ready** - Industry-standard analysis and feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- FFmpeg (for M4A support)

### Backend Setup
```bash
# Clone and setup
git clone [repo-url]
cd BeatWizard/Cursor_Pro_BeatWizard

# Create virtual environment
python -m venv beatwizard-env
source beatwizard-env/bin/activate  # On Windows: beatwizard-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run backend
python app-minimal.py
```

### Frontend Setup
```bash
# Navigate to frontend
cd beatwizard-frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 🌐 Live Deployment

- **Backend**: https://beatwizard-api.onrender.com
- **Frontend**: Coming soon...

## 📚 Documentation

- [🚀 Deployment Guide](./docs/deployment.md)
- [🎵 API Documentation](./docs/api.md)
- [🧙‍♂️ AI Features](./docs/ai-features.md)
- [📱 Frontend Guide](./beatwizard-frontend/README.md)
- [🛠️ Development Setup](./docs/development.md)

## 🏗️ Architecture

```
BeatWizard/
├── 🎵 Backend (Python/Flask)
│   ├── Audio Analysis Engine
│   ├── AI/NLU Conversation System
│   └── RESTful API
├── 🎨 Frontend (React/TypeScript)
│   ├── File Upload Interface
│   ├── AI Chat Interface
│   └── Analysis Results Display
└── 🗄️ Database (Supabase)
    ├── User Management
    ├── Analysis History
    └── Chat Sessions
```

## 🎯 Project Status

- ✅ Core audio analysis engine
- ✅ AI wizard conversation system
- ✅ Mobile-optimized frontend
- ✅ Production deployment
- 🚧 Enhanced M4A support
- 📋 User dashboard & library

## 🤝 Contributing

See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

---

**Built with ❤️ for music producers worldwide** 🌍✨
