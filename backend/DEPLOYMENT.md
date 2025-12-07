# Railway Deployment Guide

This guide walks you through deploying the Emotion Detection API backend to Railway with cost-free configuration and budget monitoring.

## Prerequisites

- GitHub account with the repository pushed
- Railway account (sign up at [railway.app](https://railway.app))
- Docker image builds successfully locally
- **Hugging Face account** (sign up at [huggingface.co](https://huggingface.co)) for model hosting

## Step 1: Prepare Your Repository

### 1.1 Upload Model to Hugging Face Hub

The model file (`emotion_classifier.onnx`) is hosted on Hugging Face Hub. First, upload it:

**Option 1: Using the provided script**

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Install huggingface CLI by following these instructructions https://huggingface.co/docs/huggingface_hub/main/en/guides/cli

# Login to Hugging Face (if not already logged in)
hf auth login

# Upload the model
./scripts/upload_model_to_hf.sh dwest1507/emotion-detection-model models/emotion_classifier.onnx
```

**Option 2: Manual upload via web interface**

1. Go to [huggingface.co](https://huggingface.co) and sign in
2. Create a new model repository: `dwest1507/emotion-detection-model`
3. Upload `emotion_classifier.onnx` to the repository
4. Make the repository public (or keep it private and use HF_TOKEN in Railway)

### 1.2 Push Code to GitHub

Ensure your backend code is pushed to GitHub:

```bash
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

**Note**: The model file is downloaded from Hugging Face Hub during the Docker build, so it doesn't need to be in the repository.

## Step 2: Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your GitHub account (if first time)
5. Select your repository containing the backend code
6. Railway will automatically detect the `Dockerfile` at the repository root

## Step 3: Configure Project Settings

### 3.1 Environment Variables

Railway will automatically set the `PORT` environment variable. The Dockerfile is configured to use it.

Optional build arguments you can set (if you want to use a different model):
- `HF_MODEL_ID` - Hugging Face model repository ID (default: `dwest1507/emotion-detection-model`)
- `MODEL_FILENAME` - Model filename (default: `emotion_classifier.onnx`)

Optional environment variables you can set (if you want to customize):
- `CONFIDENCE_THRESHOLD=0.6` (default in code)
- `MAX_FILE_SIZE_MB=5` (default in code)
- `HF_TOKEN` - Hugging Face token (only needed for private model repositories)

To set environment variables:
1. Go to your service in Railway dashboard
2. Click **Variables** tab
3. Add any custom variables if needed

## Step 4: Set Up Budget Monitoring (CRITICAL for Cost-Free Deployment)

Railway provides **$5 monthly credit** on the free tier. To ensure cost-free operation:

### 4.1 Set Usage Limit

1. Go to **Project Settings** → **Usage**
2. Click **"Set Usage Limit"**
3. Set limit to **$5.00/month**
4. Enable **"Pause project when limit is reached"** ✅
5. Click **Save**

### 4.2 Configure Email Alerts

Set up alerts to monitor usage:
1. In the **Usage** section, click **"Configure Alerts"**
2. Add email alerts at:
   - **50%** ($2.50) - Early warning
   - **80%** ($4.00) - Approaching limit
   - **90%** ($4.50) - Critical warning
3. Save alerts

### 4.3 Understanding Railway Free Tier

- **Monthly Credit**: $5.00
- **Execution Time**: 500 hours/month (~16.6 hours/day)
- **RAM**: 512 MB
- **Disk Space**: 1 GB
- **Auto-sleep**: After 15 minutes of inactivity (cold start ~20-30s)

**Expected Costs:**
- Low traffic: **$0/month** (stays within free credit)
- Moderate traffic: **$2-3/month** (well under $5 limit)
- With auto-pause enabled, project will stop before exceeding budget

## Step 5: Deploy

1. Railway will automatically start building when you connect the repository
2. Monitor the build logs in the Railway dashboard
3. First deployment takes **5-10 minutes** (building Docker image)
4. Once deployed, Railway provides a public URL like:
   ```
   https://emotion-api-production.up.railway.app
   ```

## Step 6: Verify Deployment

### 6.1 Check Health Endpoint

```bash
curl https://YOUR-RAILWAY-URL.up.railway.app/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 6.2 Check API Documentation

Visit the interactive API docs:
```
https://YOUR-RAILWAY-URL.up.railway.app/docs
```

### 6.3 Test Prediction Endpoint

Use the provided test script:
```bash
cd backend
python test_deployed_api.py --url https://YOUR-RAILWAY-URL.up.railway.app
```

## Step 7: Configure Custom Domain (Optional)

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"** or **"Add Custom Domain"**
3. Follow Railway's instructions for DNS configuration

## Monitoring and Logs

### View Logs

1. Go to your service in Railway dashboard
2. Click **"Deployments"** tab
3. Click on a deployment to see logs
4. Use **"View Logs"** for real-time log streaming

### Monitor Usage

1. Go to **Project Settings** → **Usage**
2. View current month's usage
3. Check spending trends
4. Review alert history

## Troubleshooting

### Build Fails

- Check build logs in Railway dashboard
- Verify Dockerfile is in correct location
- Ensure all dependencies are in `requirements.txt`
- **Model file not found error**: 
  - Verify the model is uploaded to Hugging Face Hub: https://huggingface.co/dwest1507/emotion-detection-model
  - Check that the repository is public (or HF_TOKEN is set in Railway if private)
  - Verify the model filename matches: `emotion_classifier.onnx`
  - Check Railway build logs for Hugging Face download errors
  - If using a different model repository, set `HF_MODEL_ID` build argument in Railway

### Application Crashes

- Check application logs
- Verify model files are present (`models/emotion_classifier.onnx`)
- Check memory usage (free tier has 512MB limit)
- Verify PORT environment variable is being used

### Cold Start Issues

- First request after inactivity takes 20-30 seconds
- This is normal for Railway's free tier auto-sleep
- Consider implementing a health check ping service to keep it warm (optional)

### Budget Concerns

- Check usage dashboard regularly
- Ensure auto-pause is enabled
- Review email alerts
- If approaching limit, consider:
  - Optimizing model size
  - Reducing request frequency
  - Using alternative platform (Render, Fly.io)

## Cost Optimization Tips

1. **Enable Auto-Sleep**: Let Railway sleep the service when idle
2. **Monitor Usage**: Check dashboard weekly
3. **Optimize Model**: Use ONNX for efficient CPU inference (already done)
4. **Set Alerts**: Get notified before reaching limits
5. **Use Free Tier Limits**: Stay within 500 hours/month execution time

## Next Steps

After successful deployment:

1. Update frontend with the Railway API URL
2. Test end-to-end functionality
3. Set up uptime monitoring (optional, e.g., UptimeRobot)
4. Document the API URL in README.md

## Support

- Railway Documentation: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Project Issues: Open an issue on GitHub

---

**Remember**: With budget monitoring enabled and auto-pause configured, your deployment will remain cost-free as long as usage stays within the $5 monthly credit.

