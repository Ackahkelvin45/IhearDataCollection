# I Hear Data Collection System

A comprehensive Django-based audio data collection, management, and analysis platform. This system enables researchers and data scientists to collect, store, process, and analyze audio datasets with AI-powered insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Development](#development)
- [API Documentation](#api-documentation)
- [Database](#database)
- [Background Tasks](#background-tasks)
- [Storage](#storage)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The I Hear Data Collection System is designed for:
- **Audio Dataset Management**: Upload, organize, and manage audio files with rich metadata
- **Audio Processing**: Automatic feature extraction (spectral, temporal, frequency analysis)
- **AI-Powered Insights**: Natural language queries and data analysis using LangChain/LangGraph
- **Data Export**: Export datasets in CSV/Excel formats with advanced filtering
- **User Management**: Custom authentication with role-based access control
- **Approval Workflows**: Dataset approval and review processes
- **Reporting**: Generate reports and visualizations

## ✨ Features

- **Audio File Management**
  - Bulk upload with chunked processing
  - Automatic audio feature extraction (MFCC, spectral features, energy metrics)
  - Audio visualization (waveforms, spectrograms, frequency analysis)
  - Support for multiple audio formats (MP3, M4A, WAV, etc.)

- **Data Organization**
  - Hierarchical classification (Category → Class → Subclass)
  - Geographic organization (Region → Community)
  - Metadata tracking (recording device, date, collector, etc.)

- **AI Data Insights**
  - Natural language SQL query generation
  - Automated data analysis and visualization recommendations
  - Chat-based interface for data exploration
  - Query caching for performance

- **Export & Reporting**
  - Paginated API for large dataset exports
  - CSV/Excel export with filtering
  - Performance-optimized raw SQL queries

- **Background Processing**
  - Celery-based task queue for audio processing
  - Chunked file uploads
  - Progress tracking and cancellation support

## 🛠 Tech Stack

### Backend
- **Django 5.2.2** - Web framework
- **Django REST Framework** - API framework
- **PostgreSQL** - Primary database
- **Redis** - Caching and Celery broker
- **Celery** - Background task processing
- **LangChain/LangGraph** - AI agent framework
- **OpenAI API** - LLM integration

### Audio Processing
- **librosa** - Audio analysis and feature extraction
- **ffmpeg** - Audio format conversion
- **numpy/scipy** - Numerical computations
- **soundfile** - Audio file I/O

### Frontend/Admin
- **Django Unfold** - Modern admin interface
- **Tailwind CSS** - Styling
- **Plotly** - Data visualization

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **Nginx** - Reverse proxy
- **Gunicorn/Uvicorn** - ASGI/WSGI server
- **DigitalOcean Spaces** (optional) - Object storage

## 📁 Project Structure

```
datacollection/
├── authentication/          # Custom user authentication
│   ├── models.py           # CustomUser model
│   ├── views.py            # Login/logout views
│   └── urls.py
├── core/                   # Core models and utilities
│   ├── models.py           # Category, Class, Subclass, Region, Community
│   ├── admin.py            # Admin configurations
│   └── management/         # Custom management commands
├── data/                   # Main data collection app
│   ├── models.py           # NoiseDataset, AudioFeature, NoiseAnalysis
│   ├── views.py            # Dataset views, export APIs
│   ├── tasks.py            # Celery tasks for audio processing
│   ├── forms.py            # Dataset forms
│   ├── serializers.py      # DRF serializers
│   ├── audio_processing.py # Audio feature extraction
│   └── urls.py
├── data_insights/          # AI-powered data insights
│   ├── workflows/          # LangGraph agent workflows
│   │   ├── agent_workflow.py
│   │   ├── sql_agent.py
│   │   ├── tools.py
│   │   └── prompt.py
│   ├── models.py           # ChatSession, QueryCache
│   └── views.py
├── approval/               # Dataset approval workflows
├── reports/                # Reporting functionality
├── datacollection/         # Project settings
│   ├── settings.py         # Django settings
│   ├── urls.py             # Root URL configuration
│   ├── celery.py           # Celery configuration
│   └── wsgi.py/asgi.py
├── docker-compose.yml      # Docker services configuration
├── Dockerfile             # Application container
├── requirements.txt       # Python dependencies
└── deploy.sh              # Deployment script
```

## 📦 Prerequisites

- **Python 3.13+**
- **PostgreSQL 16+**
- **Redis 7+**
- **Docker & Docker Compose** (for containerized deployment)
- **FFmpeg** (for audio processing)
- **Node.js & npm** (for Tailwind CSS)

## 🚀 Installation

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd datacollection
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Deploy using the script**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

   Or manually:
   ```bash
   docker compose up -d
   ```

4. **Access the application**
   - Web: http://localhost:8000
   - Admin: http://localhost:8000/admin
   - API Docs: http://localhost:8000/api/docs

### Option 2: Local Development

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y postgresql postgresql-contrib redis-server ffmpeg
   
   # On macOS
   brew install postgresql redis ffmpeg
   ```

4. **Set up PostgreSQL**
   ```bash
   sudo -u postgres psql
   CREATE DATABASE iheardatadb;
   CREATE USER postgres WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE iheardatadb TO postgres;
   ```

5. **Run migrations**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. **Start Redis**
   ```bash
   redis-server
   ```

7. **Start Celery worker** (in a separate terminal)
   ```bash
   celery -A datacollection worker -l INFO
   ```

8. **Run development server**
   ```bash
   python manage.py runserver
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database
POSTGRES_DB=iheardatadb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
USE_SQLITE=False  # Set to True for local SQLite (not recommended)

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_USE_TLS=False
REDIS_USERNAME=default

# Celery
CELERY_BROKER_DB_ID=2
CELERY_RESULT_BACKEND_DB_ID=3

# Storage (Optional - DigitalOcean Spaces)
USE_S3=False
DO_SPACES_KEY=your_spaces_key
DO_SPACES_SECRET=your_spaces_secret
DO_SPACES_BUCKET=your_bucket_name
DO_SPACES_ENDPOINT=https://lon1.digitaloceanspaces.com

# OpenAI (for AI insights)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# Email (Optional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
DEFAULT_FROM_EMAIL=your_email@gmail.com
USE_SMTP_EMAIL=True

# Security
SECRET_KEY=your_secret_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Uploads
SHARED_UPLOADS_DIR=/shared_uploads
```

### Key Settings

- **USE_S3**: Set to `True` to use DigitalOcean Spaces for media storage
- **USE_SQLITE**: Set to `True` for local development with SQLite (not recommended for production)
- **DEBUG**: Set to `False` in production
- **CELERY_TASK_ALWAYS_EAGER**: Set to `True` to run tasks synchronously (useful for testing)

## 🏃 Running the Application

### Development Mode

```bash
# Terminal 1: Django server
python manage.py runserver

# Terminal 2: Celery worker
celery -A datacollection worker -l DEBUG --concurrency 1

# Terminal 3: Celery beat (if using scheduled tasks)
celery -A datacollection beat -l INFO
```

### Production Mode (Docker)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f web
docker compose logs -f celery-worker

# Stop services
docker compose down
```

## 💻 Development

### Code Style

The project uses:
- **Black** for code formatting
- **Pre-commit hooks** for code quality

```bash
# Format code
black .

# Run pre-commit hooks
pre-commit run --all-files
```

### Database Migrations

```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# In Docker
docker compose run --rm web python manage.py migrate
```

### Creating Superuser

```bash
python manage.py createsuperuser

# In Docker
docker compose run --rm web python manage.py createsuperuser
```

### Static Files

```bash
# Collect static files
python manage.py collectstatic --noinput

# In Docker (handled automatically by migration service)
```

### Management Commands

```bash
# Seed database (if available)
python manage.py seed_db

# Wait for database (useful in Docker)
python manage.py wait_for_db
```

## 📡 API Documentation

### REST API Endpoints

The API is documented using DRF Spectacular (OpenAPI):

- **Swagger UI**: http://localhost:8000/api/docs
- **Schema**: http://localhost:8000/api/schema/

### Key API Endpoints

#### Data Export API
```
GET /api/export-data/
Query Parameters:
  - batch_size: Number of records per batch (default: 500, max: 1000)
  - offset: Starting position (default: 0)
  - search: Search query (name, noise_id, description)
  - category: Filter by category ID
  - region: Filter by region ID
  - community: Filter by community ID
  - dataset_type: Filter by dataset type ID
  - collector: Filter by collector ID

Response:
{
  "results": [...],
  "total": 1000,
  "offset": 0,
  "batch_size": 500,
  "has_more": true
}
```

#### Plot APIs
```
GET /api/dataset/<dataset_id>/plot/waveform/
GET /api/dataset/<dataset_id>/plot/spectrogram/
GET /api/dataset/<dataset_id>/plot/mfcc/
GET /api/dataset/<dataset_id>/plot/freq-features/
```

#### Bulk Upload API
```
POST /api/upload-chunk/
GET /api/bulk-upload-progress/<bulk_upload_id>/
POST /api/cancel-upload/<bulk_upload_id>/
```

### Authentication

Most API endpoints require authentication. Use Django's session authentication or token authentication:

```python
# Session auth (for web)
# Login via /auth/login/ first

# Token auth (for API clients)
# Include token in header: Authorization: Token <your-token>
```

## 🗄️ Database

### Models Overview

#### Core Models (`core/models.py`)
- **Region**: Geographic regions
- **Community**: Communities within regions
- **Category**: Dataset categories
- **Class**: Classes within categories
- **SubClass**: Subclasses within classes
- **Microphone_Type**: Recording device types
- **Time_Of_Day**: Recording time periods

#### Data Models (`data/models.py`)
- **Dataset**: Dataset type definitions
- **NoiseDataset**: Main audio dataset model
- **AudioFeature**: Extracted audio features (spectral, temporal)
- **NoiseAnalysis**: Noise analysis results (decibel levels, frequency analysis)
- **BulkAudioUpload**: Bulk upload tracking

#### Authentication (`authentication/models.py`)
- **CustomUser**: Extended user model with speaker_id

### Database Queries

The application uses both Django ORM and raw SQL for performance:

```python
# Example: Optimized export query (raw SQL)
# See data/views.py ExportDataAPIView for reference
```

## 🔄 Background Tasks

### Celery Configuration

Tasks are defined in `data/tasks.py`:

- **`process_bulk_upload`**: Process uploaded audio files
- **`extract_audio_features`**: Extract audio features from files
- **`process_audio_file`**: Complete audio processing pipeline

### Running Celery

```bash
# Development
celery -A datacollection worker -l DEBUG --concurrency 1

# Production (Docker)
# Automatically started via docker-compose.yml
```

### Task Monitoring

- **Celery Results**: Stored in Redis
- **Task Status**: Check via Django admin or API
- **Progress Tracking**: Available for bulk uploads

## 📦 Storage

### Local Storage (Default)

Media files are stored in:
- `media/` - User uploads
- `staticfiles/` - Collected static files
- `shared_uploads/` - Temporary upload assembly directory

### S3/DigitalOcean Spaces (Optional)

To use cloud storage:

1. Set `USE_S3=True` in `.env`
2. Configure DigitalOcean Spaces credentials
3. Files will be automatically uploaded to Spaces

## 🚢 Deployment

### Docker Deployment

The project includes a deployment script:

```bash
./deploy.sh
```

This script:
1. Builds Docker images
2. Starts database and Redis
3. Runs migrations and collects static files
4. Starts web, celery, and nginx services

### Manual Deployment

```bash
# Build images
docker compose build

# Start dependencies
docker compose up -d db redis

# Run migrations
docker compose run --rm migration

# Start services
docker compose up -d web celery-worker nginx
```

### Production Checklist

- [ ] Set `DEBUG=False`
- [ ] Configure proper `ALLOWED_HOSTS`
- [ ] Set secure `SECRET_KEY`
- [ ] Configure SSL/TLS (via Nginx)
- [ ] Set up proper database backups
- [ ] Configure monitoring and logging
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Configure email settings
- [ ] Review security settings
- [ ] Set up CI/CD pipeline

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection settings in .env
# Verify POSTGRES_HOST, POSTGRES_PORT, credentials
```

#### Redis Connection Errors
```bash
# Check Redis is running
redis-cli ping
# Should return: PONG

# Check Redis configuration in settings.py
```

#### Celery Not Processing Tasks
```bash
# Check Celery worker is running
celery -A datacollection inspect active

# Check Redis connection for broker
# Verify CELERY_BROKER_URL in settings
```

#### Audio Processing Failures
```bash
# Verify FFmpeg is installed
ffmpeg -version

# Check audio file formats are supported
# Check file permissions in media/ directory
```

#### Static Files Not Loading
```bash
# Collect static files
python manage.py collectstatic --noinput

# Check STATIC_ROOT and STATIC_URL settings
# Verify Nginx configuration for static files
```

### Debugging

#### View Logs (Docker)
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f web
docker compose logs -f celery-worker
```

#### Django Debug Toolbar
Add to `INSTALLED_APPS` for development debugging.

#### Database Queries
Enable query logging in `settings.py`:
```python
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django.db.backends': {
            'level': 'DEBUG',
            'handlers': ['console'],
        },
    },
}
```

## 📚 Additional Resources

### Key Files to Understand

- **`data/views.py`**: Main dataset views and export API
- **`data/tasks.py`**: Celery background tasks
- **`data/audio_processing.py`**: Audio feature extraction
- **`data_insights/workflows/agent_workflow.py`**: AI agent implementation
- **`datacollection/settings.py`**: All configuration
- **`docker-compose.yml`**: Service definitions

### External Documentation

- [Django Documentation](https://docs.djangoproject.com/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [librosa Documentation](https://librosa.org/doc/latest/)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📝 License

[Specify your license here]

## 👥 Team

[Add team information]

---

**Need Help?** Open an issue or contact the development team.

