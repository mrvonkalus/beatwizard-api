# 🚀 **BeatWizard Railway Deployment Guide**

Complete step-by-step guide to deploy BeatWizard to Railway cloud platform.

## 📋 **Prerequisites**

- ✅ GitHub account
- ✅ Railway account (free tier available)
- ✅ OpenAI API key (for intelligent feedback)
- ✅ Git installed locally

---

## 🏗️ **Step 1: Prepare Your Repository**

### **Initialize Git (if not already done)**
```bash
# In your BeatWizard project directory
git init
git add .
git commit -m "Initial BeatWizard commit - ready for Railway deployment"
```

### **Create GitHub Repository**
1. Go to [GitHub.com](https://github.com) and create a new repository
2. Name it `beatwizard-api` (or your preferred name)
3. **Don't initialize** with README, .gitignore, or license (we already have these)
4. Copy the repository URL

### **Push to GitHub**
```bash
# Replace with your actual GitHub repository URL
git remote add origin https://github.com/yourusername/beatwizard-api.git
git branch -M main
git push -u origin main
```

---

## ☁️ **Step 2: Deploy to Railway**

### **Sign Up for Railway**
1. Go to [Railway.app](https://railway.app)
2. Sign up using your GitHub account
3. Verify your account

### **Create New Project**
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your `beatwizard-api` repository
4. Railway will automatically detect it's a Python project

### **Configure Environment Variables**
In your Railway project dashboard:

1. Click **"Variables"** tab
2. Add these environment variables:

**Required Variables:**
```
SECRET_KEY=your-super-secret-key-change-this-in-production-123456789
FLASK_ENV=production
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO
CORS_ORIGINS=*
```

**OpenAI Integration (Recommended):**
```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Optional (for custom domain):**
```
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
```

### **Deploy**
1. Railway will automatically start building and deploying
2. Monitor the **"Deployments"** tab for build progress
3. First deployment may take 5-10 minutes (installing audio libraries)

---

## 🔍 **Step 3: Verify Deployment**

### **Get Your App URL**
1. In Railway dashboard, click **"Settings"**
2. Click **"Generate Domain"** to get a public URL
3. Your URL will be something like: `https://beatwizard-api-production.railway.app`

### **Test Your Deployment**
```bash
# Test the health endpoint
curl https://your-app-url.railway.app/health

# Test supported formats
curl https://your-app-url.railway.app/api/formats
```

### **Use the Test Script**
```bash
# Update the URL in test_api.py and run:
python test_api.py
```

---

## 📊 **Step 4: Monitor Your Deployment**

### **View Logs**
- In Railway dashboard, click **"Deployments"**
- Click on your latest deployment
- View **"Build Logs"** and **"Deploy Logs"**

### **Monitor Performance**
- Railway provides CPU, Memory, and Network usage metrics
- Set up alerts if needed

### **Health Checks**
Railway automatically monitors: `https://your-app.railway.app/health`

---

## 🔧 **Step 5: Frontend Integration**

### **API Endpoints Available:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Basic health check |
| `/health` | GET | Detailed health check |
| `/api/upload` | POST | Upload & analyze audio file |
| `/api/formats` | GET | Get supported formats |

### **Frontend Configuration:**
```javascript
// In your frontend .env
REACT_APP_API_URL=https://your-app-url.railway.app

// Example API call
const response = await fetch(`${process.env.REACT_APP_API_URL}/api/upload`, {
  method: 'POST',
  body: formData
});
```

---

## 🚨 **Troubleshooting**

### **Common Issues:**

**1. Build Fails - Audio Library Installation**
```
Solution: Check railway.json build command includes all dependencies
```

**2. Memory Issues**
```
Solution: Upgrade Railway plan or optimize analyzer settings
```

**3. Timeout Errors**
```
Solution: Increase timeout in railway.json (currently 300s)
```

**4. CORS Errors**
```
Solution: Update CORS_ORIGINS environment variable
```

### **Debug Commands:**
```bash
# Check deployment logs
railway logs

# SSH into container (if needed)
railway shell

# Redeploy
git push origin main
```

---

## 💰 **Cost Optimization**

### **Railway Free Tier:**
- ✅ 512MB RAM, 1 vCPU
- ✅ $5 free credits per month
- ✅ Perfect for testing and low traffic

### **Production Recommendations:**
- **Starter Plan**: $5/month for higher limits
- **Pro Plan**: $20/month for production use
- Monitor usage in Railway dashboard

---

## 🔐 **Security Best Practices**

### **Environment Variables:**
- ✅ Never commit API keys to GitHub
- ✅ Use strong SECRET_KEY
- ✅ Limit CORS origins to your domain

### **API Security:**
- ✅ Rate limiting (consider adding)
- ✅ File size limits (200MB currently)
- ✅ File type validation

---

## 📈 **Scaling & Updates**

### **Auto-Deployment:**
- ✅ Every `git push` triggers new deployment
- ✅ Zero-downtime deployments
- ✅ Automatic rollback on failure

### **Performance Monitoring:**
```bash
# Add to your frontend
fetch('/health').then(r => r.json()).then(data => {
  console.log('API Status:', data.status);
});
```

### **Database Integration (Future):**
```
1. Add PostgreSQL service in Railway
2. Update DATABASE_URL environment variable
3. Store analysis history and user data
```

---

## ✅ **Deployment Checklist**

- [ ] Repository pushed to GitHub
- [ ] Railway project created
- [ ] Environment variables configured
- [ ] First deployment successful
- [ ] Health check passing
- [ ] API endpoints tested
- [ ] Frontend integration working
- [ ] Domain configured (optional)
- [ ] Monitoring setup

---

## 🎯 **Next Steps**

1. **Add Database**: PostgreSQL for user data and analysis history
2. **Custom Domain**: Point your domain to Railway
3. **CDN**: Add CloudFlare for faster global access
4. **Monitoring**: Set up error tracking (Sentry)
5. **Analytics**: Track usage and performance

---

## 📞 **Support**

**Railway Issues:**
- [Railway Documentation](https://docs.railway.app)
- [Railway Discord](https://discord.gg/railway)

**BeatWizard Issues:**
- Check deployment logs
- Test locally first
- Verify environment variables

---

## 🚀 **Quick Start Commands**

```bash
# Complete deployment in 5 commands:
git add .
git commit -m "Deploy to Railway"
git push origin main
# Go to Railway dashboard and add environment variables
# Test: curl https://your-app.railway.app/health
```

**Your BeatWizard API is now live and ready for production use!** 🎵✨