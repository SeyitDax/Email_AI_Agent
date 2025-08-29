# 🤖 AI Email Agent

A sophisticated, production-ready AI system for automated customer support email processing using LangChain and OpenAI. This system intelligently classifies emails, generates appropriate responses, and handles complex escalation scenarios with advanced decision-making algorithms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)

## 🌟 Features

### Advanced AI Classification System
- **19 Category Classification**: Sophisticated multi-factor email categorization
- **Sentiment Analysis**: Advanced emotional intelligence with contextual understanding
- **Confidence Scoring**: Multi-dimensional confidence assessment for reliable decision making
- **Complexity Analysis**: Intelligent complexity evaluation for appropriate handling

### Smart Escalation Engine
- **Intelligent Routing**: Context-aware team assignment (Management, Technical, Billing, Partnerships)
- **Priority Management**: Dynamic priority levels based on urgency and business impact
- **Legal Threat Detection**: Automatic identification of legal concerns for management attention
- **VIP Customer Handling**: Special processing for high-value customers

### Enhanced Email Categories
- **Customer Praise**: Thank you messages and positive feedback
- **Feature Suggestions**: Enhancement requests and improvement ideas
- **Partnership Business**: Business inquiries and collaboration requests
- **Subscription Management**: Account and billing management requests
- **Technical Support**: Bug reports and technical assistance
- **Billing Inquiries**: Payment and invoice related questions

### Production-Ready Architecture
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Streamlit Dashboard**: Real-time email processing interface
- **Docker Support**: Complete containerization for easy deployment
- **Comprehensive Testing**: Unit tests, integration tests, and performance benchmarks
- **Error Handling**: Robust error management with detailed logging

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/seyitdax/email-ai-agent.git
   cd email-ai-agent
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

### Running the Application

#### Streamlit Dashboard
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`

#### FastAPI Backend
```bash
uvicorn src.api.main:app --reload
```
- API Documentation: `http://localhost:8000/docs`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

#### Docker Deployment
```bash
docker-compose up -d
```

## 🏗️ Architecture

### Core Components

```
email-ai-agent/
├── src/
│   ├── agents/               # AI Processing Components
│   │   ├── classifier.py     # 19-category email classification
│   │   ├── confidence_scorer.py  # Multi-dimensional confidence analysis
│   │   ├── escalation_engine.py  # Smart escalation decision system
│   │   ├── responder.py      # Response generation and orchestration
│   │   └── exchange_handler.py   # Return/exchange request processing
│   ├── api/                  # FastAPI Application
│   │   ├── main.py          # Main FastAPI app
│   │   └── routes/          # API endpoints
│   ├── core/                # Core Configuration
│   │   ├── config.py        # Application settings
│   │   ├── models.py        # Data models
│   │   └── exceptions.py    # Custom exceptions
│   └── services/            # External Services
│       ├── database.py      # Database operations
│       └── email_service.py # Email handling
├── tests/                   # Comprehensive Test Suite
├── docker/                  # Docker configuration
├── app.py                   # Streamlit dashboard
└── email_agent.py          # Legacy compatibility wrapper
```

### AI Processing Pipeline

1. **Email Reception** → Raw email content input
2. **Classification** → 19-category classification with confidence scoring
3. **Sentiment Analysis** → Emotional tone assessment
4. **Escalation Analysis** → Smart routing decision
5. **Response Generation** → Context-aware response creation
6. **Quality Assessment** → Response quality evaluation

## 📊 Classification Categories

### Customer Service Categories
- **Customer Support**: General inquiries and assistance requests
- **Technical Issue**: Bug reports, system problems, and technical difficulties
- **Billing Inquiry**: Payment questions, invoice issues, account billing
- **Refund Request**: Return and refund processing
- **Product Question**: Product information and feature inquiries

### Enhanced Categories (New!)
- **Customer Praise**: Thank you messages and positive feedback
- **Feature Suggestions**: Enhancement requests and improvement ideas
- **Partnership Business**: Business development and collaboration inquiries
- **Subscription Management**: Account modifications and billing updates

### Specialized Categories
- **VIP Customer**: High-value customer communications
- **Legal Compliance**: Legal notices and compliance issues
- **Press Media**: Media inquiries and press requests
- **Urgent High Priority**: Time-sensitive communications

## 🎯 Escalation Logic

### Smart Team Assignment
- **Management**: Legal threats, service crises, business critical issues
- **Technical Team**: Software bugs, system issues, technical support
- **Billing Specialists**: Payment issues, account billing, subscriptions
- **Partnerships**: Business development, collaboration opportunities
- **VIP Concierge**: High-value customer support
- **Legal Compliance**: Regulatory and legal matters

### Priority Levels
- **Critical (1)**: Immediate response required (< 1 hour)
- **High (2)**: Same-day response (< 4 hours)
- **Medium (3)**: Next business day (< 24 hours)
- **Low (4)**: Standard SLA (< 48 hours)
- **Routine (5)**: Normal queue processing (< 72 hours)

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# All tests with coverage
python -m pytest --cov=src tests/
```

### Performance Benchmarks
```bash
python -m pytest tests/performance/ -v
```

### Specific Test Categories
```bash
# Test classifier accuracy
python -m pytest tests/unit/test_classifier.py -v

# Test escalation logic
python -m pytest tests/unit/test_escalation_engine.py -v
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database (if using)
DATABASE_URL=sqlite:///./email_agent.db

# Redis (for caching)
REDIS_URL=redis://localhost:6379

# Email Service Settings
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_email_password
```

### Configuration Options

The system supports extensive configuration through `src/core/config.py`:

- **Confidence Thresholds**: Adjustable confidence levels for escalation
- **Classification Settings**: Category weights and scoring parameters  
- **Response Templates**: Customizable response templates
- **Escalation Rules**: Business-specific escalation logic
- **Rate Limiting**: API call limits and caching settings

## 📈 Performance

### Benchmarks
- **Classification Speed**: ~50ms per email
- **Response Generation**: ~200ms per response
- **Throughput**: 100+ emails/minute
- **Accuracy**: 95%+ classification accuracy
- **Confidence**: Average 85%+ confidence scores

### Optimization Features
- **Intelligent Caching**: Response and classification caching
- **Async Processing**: Non-blocking email processing
- **Batch Operations**: Efficient bulk email handling
- **Connection Pooling**: Optimized database connections

## 🛡️ Security

### Security Features
- **Input Validation**: Comprehensive input sanitization
- **API Key Management**: Secure credential handling
- **Rate Limiting**: Protection against abuse
- **Error Handling**: Safe error responses without information leakage
- **Logging**: Comprehensive audit trails

### Best Practices
- Store API keys in environment variables
- Use HTTPS in production
- Regularly rotate API keys
- Monitor API usage and costs
- Implement proper authentication for API endpoints

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Scale services
docker-compose up -d --scale api=3
```

### Production Deployment
1. **Environment Setup**
   ```bash
   export ENVIRONMENT=production
   export DEBUG=false
   export OPENAI_API_KEY=your_production_key
   ```

2. **Database Migration** (if applicable)
   ```bash
   alembic upgrade head
   ```

3. **Start Services**
   ```bash
   gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

### Monitoring
- **Health Checks**: `/health` endpoint for service monitoring
- **Metrics**: Detailed processing metrics and performance data
- **Logging**: Structured logging with multiple levels
- **Error Tracking**: Comprehensive error reporting

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -r requirements-dev.txt`)
4. Run tests (`python -m pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

### Testing Requirements
- All tests must pass
- Code coverage should be > 90%
- Include integration tests for new features
- Performance benchmarks for critical paths

## 📚 Documentation

### API Documentation
- **Interactive Docs**: Available at `/docs` when running FastAPI
- **OpenAPI Schema**: Available at `/openapi.json`
- **Postman Collection**: Available in `docs/api/`

### Learning Resources
- [LangChain Documentation](https://docs.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🐛 Troubleshooting

### Common Issues

#### OpenAI API Errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connection
python -c "import openai; print(openai.Model.list())"
```

#### Classification Issues
- **Low Confidence**: Check email content length and clarity
- **Wrong Category**: Review classification keywords and patterns
- **Performance**: Enable caching for repeated classifications

#### Escalation Problems
- **Wrong Team Assignment**: Review escalation patterns and weights
- **Incorrect Priority**: Check sentiment analysis and risk factors
- **Missing Escalations**: Verify escalation thresholds

### Getting Help
1. Check the [Issues](https://github.com/seyitdax/email-ai-agent/issues) page
2. Review the troubleshooting guide
3. Enable debug logging for detailed information
4. Contact support with detailed error logs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT models
- **LangChain** for the excellent framework
- **FastAPI** for the high-performance web framework
- **Streamlit** for the intuitive dashboard interface
- **Contributors** who helped improve this project

## 📞 Contact

**Seyit Ahmet Demir**
- GitHub: [@seyitdax](https://github.com/seyitdax)
- Email: seyitdax@gmail.com

---

<div align="center">
  <p><strong>Built with ❤️ by Seyit Ahmet Demir</strong></p>
  <p>⭐ Star this repository if it helped you!</p>
</div>