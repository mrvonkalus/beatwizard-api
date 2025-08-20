# 🤝 Contributing to BeatWizard

Thank you for considering contributing to BeatWizard! This document provides guidelines and information for contributors.

## 🌟 Code of Conduct

BeatWizard is committed to providing a welcoming and inclusive environment for all contributors. Please be respectful, constructive, and helpful in all interactions.

## 🚀 Quick Start for Contributors

### 1. Fork & Clone
```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR-USERNAME/beatwizard-api.git
cd beatwizard-api
```

### 2. Set Up Development Environment
```bash
# Backend setup
python -m venv beatwizard-env
source beatwizard-env/bin/activate  # Windows: beatwizard-env\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Frontend setup
cd beatwizard-frontend
npm install
```

### 3. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 🎯 Types of Contributions

### 🐛 Bug Fixes
- Check existing issues first
- Create detailed bug report if none exists
- Include steps to reproduce
- Test your fix thoroughly

### ✨ New Features
- Discuss in issues before starting large features
- Follow existing code patterns
- Include tests and documentation
- Consider mobile and accessibility

### 🧙‍♂️ AI/NLU Improvements
- Understand the wizard personality system
- Test across all skill levels (beginner/intermediate/advanced)
- Ensure responses are engaging and varied
- Follow conversation flow guidelines

### 📱 Frontend/UI Enhancements
- Maintain mobile-first approach
- Follow Tailwind CSS conventions
- Test across browsers and devices
- Ensure accessibility standards

### 📚 Documentation
- Keep documentation up-to-date
- Include code examples
- Use clear, beginner-friendly language
- Add screenshots for UI features

## 🔧 Development Guidelines

### Code Style

#### Python (Backend)
```python
# Use type hints
def analyze_audio(file_path: str) -> Dict[str, Any]:
    """Analyze audio file and return results."""
    pass

# Follow PEP 8
# Use descriptive variable names
# Add docstrings for functions and classes
# Handle errors gracefully
```

#### TypeScript/React (Frontend)
```typescript
// Use TypeScript interfaces
interface AudioAnalysisResults {
  tempo: number;
  key: string;
  loudness: number;
}

// Use functional components with hooks
const AudioUpload: React.FC<Props> = ({ onAnalysis }) => {
  // Component logic
};

// Use meaningful component and variable names
// Follow React best practices
// Ensure mobile responsiveness
```

### Testing Strategy

#### Backend Testing
```bash
# Run unit tests
python -m pytest beatwizard/tests/

# Test API endpoints
python test_analysis_local.py

# Test with different audio formats
python verify_deployment.py
```

#### Frontend Testing
```bash
# Run tests
npm test

# Build and verify
npm run build

# Manual testing checklist:
# - File upload works
# - Chat interface responds
# - Mobile experience is smooth
# - All buttons and links work
```

### Performance Guidelines

#### Backend
- Keep API response times under 3 seconds
- Optimize audio processing for memory usage
- Handle large files gracefully
- Use appropriate error handling

#### Frontend
- Keep bundle size reasonable (<200KB gzipped)
- Optimize images and assets
- Use lazy loading where appropriate
- Ensure smooth animations (60fps)

## 🎵 Audio Processing Guidelines

### Supported Formats
- **Primary**: WAV, FLAC (instant processing)
- **Secondary**: MP3, OGG (good compatibility)
- **Advanced**: M4A (with fallback conversion)

### File Size Limits
- Development: 100MB max
- Production: 25MB max
- Recommend optimizing for <10MB files

### Quality Standards
- Support sample rates: 22kHz - 192kHz
- Handle mono and stereo files
- Preserve audio quality during processing
- Provide meaningful error messages

## 🧙‍♂️ AI Wizard Guidelines

### Personality Consistency
- Maintain magical, helpful personality
- Adapt language to user skill level
- Use emojis and mystical language appropriately
- Be encouraging and constructive

### Response Quality
- Provide actionable advice
- Include specific numbers and metrics
- Offer learning resources
- Ask engaging follow-up questions

### Technical Accuracy
- Ensure all technical advice is correct
- Stay current with industry standards
- Provide context for recommendations
- Link to authoritative sources when possible

## 📱 Mobile Development Guidelines

### Responsive Design
- Mobile-first approach
- Touch-friendly buttons (44px minimum)
- Readable text on small screens
- Smooth scrolling and interactions

### Performance
- Optimize for slower connections
- Minimize data usage
- Fast loading times
- Efficient animations

## 🔄 Pull Request Process

### Before Submitting
1. **Test thoroughly** on multiple devices/browsers
2. **Update documentation** if needed
3. **Add or update tests** for new features
4. **Check performance impact**
5. **Review your own code** first

### PR Requirements
- Clear, descriptive title
- Detailed description of changes
- Link to related issues
- Screenshots for UI changes
- Test results and evidence

### Review Process
1. **Automated checks** must pass
2. **Code review** by maintainer
3. **Testing** on staging environment
4. **Approval** and merge

## 🐛 Reporting Issues

### Bug Reports
Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:
- Clear reproduction steps
- Expected vs actual behavior
- Environment details
- Screenshots/error messages
- Audio file details (if relevant)

### Feature Requests
Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:
- Problem statement
- Proposed solution
- User stories
- Success metrics

## 🎯 Development Priorities

### Current Focus Areas
1. **Enhanced M4A Support** - Improving codec compatibility
2. **AI Conversation Flow** - Making wizard interactions more natural
3. **Mobile Optimization** - Perfect mobile experience
4. **Performance** - Faster analysis and response times

### Future Roadmap
- User accounts and analysis history
- Advanced audio effects and recommendations
- Collaborative features
- Mobile app development

## 🌟 Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes for significant contributions
- Hall of fame in documentation
- Special thanks in app credits

## 📞 Getting Help

### Communication Channels
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Discord** - Real-time chat and community (coming soon)

### Development Questions
- Check existing documentation first
- Search closed issues for similar problems
- Create detailed issue with context
- Tag appropriate maintainers

## 📄 License

By contributing to BeatWizard, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Thank you for making BeatWizard magical! 🧙‍♂️✨**

Happy coding! 🎵🚀
