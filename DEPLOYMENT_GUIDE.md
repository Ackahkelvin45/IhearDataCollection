# Deployment Guide - Export Optimization

This guide explains how to deploy the optimized export functionality that prevents 504 Gateway Timeout errors.

## Changes Made

### 1. Backend Optimization (`data/views.py`)
- **Replaced ORM queries with raw SQL** for maximum performance
- Uses direct database queries with LEFT JOINs to fetch all related data in a single query
- Reduced batch size to 500 records per request (from 2000) for faster response times
- Added detailed logging to track query performance
- Optimized data processing to avoid unnecessary object instantiation

### 2. Frontend Optimization (`data/templates/data/datasetlist.html`)
- **Increased parallel batch requests** from 3 to 5 for faster overall fetching
- Adjusted batch size to match backend (500 records per batch)
- Added total count parameter to subsequent requests to avoid recalculating count
- Improved error handling in fetch operations

### 3. Nginx Timeout Configuration (`nginx/iheardatacollection.conf`)
- **Increased proxy timeouts** from 60s to 300s (5 minutes):
  - `proxy_connect_timeout 300s`
  - `proxy_send_timeout 300s`
  - `proxy_read_timeout 300s`
  - `send_timeout 300s`

### 4. Uvicorn Timeout Configuration (`docker-compose.yml`)
- **Added keep-alive timeout** to uvicorn: `--timeout-keep-alive 300`

## Deployment Steps

### Step 1: Backup Current State
```bash
# Backup database (optional but recommended)
docker-compose exec db pg_dump -U postgres iheardatadb > backup_$(date +%Y%m%d).sql

# Backup current code
cp -r /path/to/datacollection /path/to/datacollection_backup_$(date +%Y%m%d)
```

### Step 2: Apply Code Changes
```bash
# Pull latest changes (if using git)
cd /Users/kelvinackah/Desktop/projects/freelance/datacollection
git pull origin main

# Or manually copy updated files:
# - data/views.py
# - data/templates/data/datasetlist.html
# - nginx/iheardatacollection.conf
# - docker-compose.yml
```

### Step 3: Rebuild and Restart Services
```bash
# Rebuild the Docker images with new code
docker-compose build web

# Restart all services to apply changes
docker-compose down
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Step 4: Verify Nginx Configuration
```bash
# Test nginx configuration
docker-compose exec nginx nginx -t

# If test passes, reload nginx
docker-compose exec nginx nginx -s reload
```

### Step 5: Monitor Logs
```bash
# Monitor web service logs for API performance
docker-compose logs -f web

# Look for log entries like:
# "Export API: offset=0, batch_size=500, total=23453"
# "Raw SQL query completed in 2.34s, fetched 500 rows"
# "Total API time: 2.45s for 500 records"
```

## Testing the Export

1. **Login to your application**: https://ihearandsee-at-rail.com/
2. **Navigate to the dataset list page**: `/datasetlist/`
3. **Click "Export to Excel" or "Export to CSV"**
4. **Monitor progress**:
   - You should see a SweetAlert2 loading indicator
   - Progress will show: "Fetching data... X of Y records (Z%)"
   - For 23,453 records with 500 per batch, expect ~47 batch requests
   - With 5 parallel requests, this should take approximately 10-15 batches (rounds)

## Expected Performance

### Before Optimization
- **Batch size**: 2000 records
- **Parallel requests**: 3
- **Query method**: Django ORM with select_related/prefetch_related
- **Result**: 504 Gateway Timeout after 2.2 minutes

### After Optimization
- **Batch size**: 500 records (faster per-request response)
- **Parallel requests**: 5 (faster overall completion)
- **Query method**: Raw SQL with single JOIN query
- **Timeouts**: 300 seconds (5 minutes) at all layers
- **Expected result**: 
  - Each batch request: 2-5 seconds
  - Total time for 23,453 records: 3-5 minutes (well within timeout limits)
  - Each parallel batch round: 2-5 seconds
  - Total batches: ~47
  - Total rounds (with 5 parallel): ~10 rounds
  - **Total time: ~20-50 seconds for full export**

## Troubleshooting

### If 504 errors persist:

1. **Check database performance**:
   ```bash
   docker-compose exec db psql -U postgres -d iheardatadb
   # Run EXPLAIN ANALYZE on the export query
   ```

2. **Add database indexes** (if missing):
   ```sql
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_noisedataset_category ON data_noisedataset(category_id);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_noisedataset_region ON data_noisedataset(region_id);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_noisedataset_community ON data_noisedataset(community_id);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audiofeature_dataset ON data_audiofeature(noise_dataset_id);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_noiseanalysis_dataset ON data_noiseanalysis(noise_dataset_id);
   ```

3. **Reduce batch size further**:
   - In `data/views.py`, change `batch_size = int(request.GET.get("batch_size", 500))` to `300`
   - In `data/templates/data/datasetlist.html`, change `const batchSize = 500;` to `300`

4. **Check resource limits**:
   ```bash
   # Check if containers are hitting memory/CPU limits
   docker stats
   
   # If needed, increase limits in docker-compose.yml:
   # web service -> resources -> limits -> memory: 2G
   ```

5. **Check application logs**:
   ```bash
   # Look for slow queries or errors
   docker-compose logs web | grep -i "export"
   docker-compose logs nginx | grep -i "timeout"
   ```

### If export is slow but doesn't timeout:

1. **Verify parallel requests are working**:
   - Open browser DevTools -> Network tab
   - You should see 5 requests running simultaneously
   - Each marked as "pending" until completed

2. **Check network conditions**:
   - Slow network will impact multiple batch fetches
   - Consider downloading directly on server and providing download link

## Rollback Plan

If issues occur, rollback using:

```bash
# Stop current services
docker-compose down

# Restore from backup
cp -r /path/to/datacollection_backup_YYYYMMDD/* /path/to/datacollection/

# Restart services
docker-compose up -d
```

## Additional Optimization Options (Future)

If export performance is still not satisfactory:

1. **Implement server-side caching**:
   - Cache export data for 5-10 minutes
   - Return cached results for repeated exports

2. **Background job processing**:
   - Use Celery to generate export file in background
   - Provide download link when ready
   - Send email notification on completion

3. **Pagination with client-side filtering**:
   - Export only current page or selection
   - Provide "Export All" as separate async job

4. **Database read replicas**:
   - Offload export queries to read replica
   - Reduce impact on main database

## Support

For issues or questions:
1. Check application logs: `docker-compose logs -f web`
2. Check nginx logs: `docker-compose logs -f nginx`
3. Verify database connectivity: `docker-compose exec web python manage.py dbshell`
4. Contact development team with log excerpts and error messages

