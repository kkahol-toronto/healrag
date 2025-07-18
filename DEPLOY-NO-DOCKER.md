# HEALRAG Deployment Without Docker

This guide provides multiple deployment options for environments where Docker Desktop is not available (VDI, restricted admin access, etc.).

## 🎯 **Option 1: Azure App Service Python Deployment (Recommended)**

Deploy directly to Azure App Service using native Python runtime - **no Docker required**.

### Prerequisites
- Azure CLI installed and logged in
- Active Azure subscription
- Your `.env` file configured with all required environment variables

### Quick Deployment

```bash
# Make the script executable (if not already)
chmod +x deploy-python.sh

# Deploy with default settings
./deploy-python.sh deploy

# Or deploy with custom settings
./deploy-python.sh deploy my-app-name my-resource-group westus2
```

### What This Does
1. ✅ Creates a ZIP package with your Python app
2. ✅ Creates Azure Resource Group (if needed)
3. ✅ Creates App Service Plan with Python 3.11 runtime
4. ✅ Creates Web App
5. ✅ Uploads your application code
6. ✅ Configures environment variables from `.env`
7. ✅ Starts the application with Gunicorn

### Monitor Deployment
```bash
# View real-time logs
./deploy-python.sh logs

# Or use Azure CLI directly
az webapp log tail --name healrag-security --resource-group medical
```

---

## 🎯 **Option 2: Azure Container Instances (ACI)**

Deploy using Azure's serverless container service - **no local Docker required**.

### Prerequisites
- Azure CLI
- Container image already built (can be done in Azure Container Registry build service)

### Steps

1. **Build in Azure Container Registry:**
```bash
# Create ACR (one time setup)
az acr create --resource-group medical --name healragregistry --sku Basic --admin-enabled true

# Build image in the cloud
az acr build --registry healragregistry --image healrag:latest .
```

2. **Deploy to Container Instances:**
```bash
# Get ACR credentials
ACR_LOGIN_SERVER=$(az acr show --name healragregistry --query loginServer --output tsv)
ACR_PASSWORD=$(az acr credential show --name healragregistry --query passwords[0].value --output tsv)

# Create container instance
az container create \
    --resource-group medical \
    --name healrag-instance \
    --image $ACR_LOGIN_SERVER/healrag:latest \
    --cpu 2 \
    --memory 4 \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username healragregistry \
    --registry-password $ACR_PASSWORD \
    --ports 8000 \
    --environment-variables \
        AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
        AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
        AZURE_OPENAI_KEY="$AZURE_OPENAI_KEY" \
        # ... add other env vars
```

---

## 🎯 **Option 3: GitHub Actions Deployment**

Use GitHub Actions to build and deploy - **all building happens in the cloud**.

### Setup

1. **Create `.github/workflows/deploy.yml`:**
```yaml
name: Deploy HEALRAG to Azure

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Azure Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: 'healrag-security'
        package: '.'
```

2. **Configure Secrets in GitHub:**
- `AZURE_CREDENTIALS` - Service principal credentials
- Add all your environment variables as GitHub secrets

---

## 🎯 **Option 4: Manual Azure Portal Deployment**

Deploy through the Azure Portal web interface - **completely browser-based**.

### Steps

1. **Create App Service in Portal:**
   - Go to Azure Portal → Create Resource → Web App
   - Choose Python 3.11 runtime
   - Create new resource group and app service plan

2. **Configure Deployment:**
   - Go to your App Service → Deployment Center
   - Choose "Local Git" or "GitHub" as source
   - Follow the setup instructions

3. **Upload Code:**
   - Use Azure Cloud Shell or local git to push code
   - Or use VS Code Azure extension to deploy

4. **Set Environment Variables:**
   - Go to Configuration → Application Settings
   - Add all variables from your `.env` file

---

## 🎯 **Option 5: Local Python Deployment (Development)**

Run locally for development without Docker.

### Setup
```bash
# Create virtual environment
python3 -m venv healrag_env
source healrag_env/bin/activate  # On Windows: healrag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the application
python start_api.py
```

### Access Application
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## 🛠️ **Troubleshooting**

### Common Issues

**Azure CLI Not Found:**
```bash
# Install Azure CLI (no admin required in many cases)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Or download installer from Microsoft
```

**Permission Issues:**
```bash
# Make scripts executable
chmod +x deploy-python.sh

# Check if you have write permissions
ls -la
```

**Environment Variables Missing:**
```bash
# Check your .env file exists and is properly formatted
cat .env | head -5

# Verify no spaces around = signs
# Format: KEY=value (not KEY = value)
```

**Deployment Fails:**
```bash
# Check Azure CLI login
az account show

# Re-login if needed
az login

# Check resource quotas in Azure Portal
```

### Getting Help

1. **View Application Logs:**
```bash
./deploy-python.sh logs
```

2. **Test Deployment:**
```bash
./deploy-python.sh test
```

3. **Clean Up:**
```bash
./deploy-python.sh clean
```

---

## 📋 **Deployment Checklist**

Before deploying, ensure:

- [ ] `.env` file is complete with all required variables
- [ ] Azure CLI is installed and logged in
- [ ] You have appropriate Azure subscription permissions
- [ ] Resource group name is available (or exists)
- [ ] App Service name is globally unique
- [ ] All required Azure services are set up (Storage, OpenAI, etc.)

---

## 🚀 **Quick Start Commands**

```bash
# Check if Azure CLI works
az account show

# Deploy everything
./deploy-python.sh deploy

# Monitor logs
./deploy-python.sh logs

# Test deployment
curl https://healrag-security.azurewebsites.net/health/simple
```

Choose the option that works best for your environment! **Option 1 (Azure App Service Python)** is recommended for most scenarios without Docker. 