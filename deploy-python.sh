#!/bin/bash

# HEALRAG Python Direct Deployment Script
# Deploy directly to Azure App Service without Docker
# Perfect for VDI environments without Docker Desktop

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
if [ "$1" = "deploy" ]; then
    APP_NAME="${2:-healrag-security}"
    RESOURCE_GROUP="${3:-medical}"
    LOCATION="${4:-eastus}"
else
    APP_NAME="${1:-healrag-security}"
    RESOURCE_GROUP="${2:-medical}"
    LOCATION="${3:-eastus}"
fi
RUNTIME="PYTHON|3.11"

echo "🐍 HEALRAG Python Direct Deployment"
echo "===================================="
echo "📱 App Name: $APP_NAME"
echo "📦 Resource Group: $RESOURCE_GROUP"
echo "📍 Location: $LOCATION"
echo "🐍 Runtime: $RUNTIME"
echo ""

# Function to validate Azure CLI
validate_azure_cli() {
    echo "🔍 Validating Azure CLI..."
    
    if ! command -v az &> /dev/null; then
        echo "❌ Azure CLI not found. Please install it first:"
        echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    
    if ! az account show > /dev/null 2>&1; then
        echo "❌ Azure CLI not logged in. Running login..."
        az login
    fi
    
    echo "✅ Azure CLI validated!"
}

# Function to create deployment package
create_deployment_package() {
    echo "📦 Creating deployment package..."
    
    # Create a temporary deployment directory
    DEPLOY_DIR="./deploy_temp"
    rm -rf $DEPLOY_DIR
    mkdir -p $DEPLOY_DIR
    
    # Copy application files
    echo "   Copying application files..."
    cp -r healraglib $DEPLOY_DIR/
    cp main.py $DEPLOY_DIR/
    cp requirements.txt $DEPLOY_DIR/
    cp start_api.py $DEPLOY_DIR/
    
    # Copy other necessary files if they exist
    [ -f rag_pipeline.py ] && cp rag_pipeline.py $DEPLOY_DIR/
    [ -f training_pipeline.py ] && cp training_pipeline.py $DEPLOY_DIR/
    [ -f setup.py ] && cp setup.py $DEPLOY_DIR/
    
    # Create startup script for Azure
    cat > $DEPLOY_DIR/startup.sh << 'EOF'
#!/bin/bash
echo "Starting HEALRAG application..."
cd /home/site/wwwroot

# Install dependencies if not cached
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start the application with Gunicorn
echo "Starting Gunicorn server..."
exec gunicorn main:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile -
EOF
    
    chmod +x $DEPLOY_DIR/startup.sh
    
    # Create ZIP package
    echo "   Creating ZIP package..."
    cd $DEPLOY_DIR
    zip -r ../healrag-deployment.zip . > /dev/null
    cd ..
    
    echo "✅ Deployment package created: healrag-deployment.zip"
}

# Function to create or update resource group
ensure_resource_group() {
    echo "🏗️  Ensuring resource group exists..."
    
    if ! az group show --name $RESOURCE_GROUP > /dev/null 2>&1; then
        echo "   Creating resource group: $RESOURCE_GROUP"
        az group create --name $RESOURCE_GROUP --location $LOCATION
    else
        echo "   Resource group already exists: $RESOURCE_GROUP"
    fi
    
    echo "✅ Resource group ready!"
}

# Function to create App Service Plan
create_app_service_plan() {
    local plan_name="$APP_NAME-plan"
    
    echo "📋 Creating App Service Plan..."
    
    if ! az appservice plan show --name $plan_name --resource-group $RESOURCE_GROUP > /dev/null 2>&1; then
        echo "   Creating new plan: $plan_name"
        az appservice plan create \
            --name $plan_name \
            --resource-group $RESOURCE_GROUP \
            --sku P1v2 \
            --is-linux \
            --location $LOCATION
    else
        echo "   Plan already exists: $plan_name"
    fi
    
    echo "✅ App Service Plan ready!"
}

# Function to create or update Web App
create_webapp() {
    local plan_name="$APP_NAME-plan"
    
    echo "🌐 Creating Web App..."
    
    if ! az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP > /dev/null 2>&1; then
        echo "   Creating new web app: $APP_NAME"
        az webapp create \
            --name $APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --plan $plan_name \
            --runtime "$RUNTIME"
    else
        echo "   Web app already exists: $APP_NAME"
    fi
    
    echo "✅ Web App ready!"
}

# Function to configure app settings
configure_app_settings() {
    echo "⚙️  Configuring application settings..."
    
    # Read .env file and set app settings
    if [ -f .env ]; then
        echo "   Setting environment variables from .env file..."
        
        # Convert .env to Azure app settings format
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ $key =~ ^#.*$ ]] && continue
            [[ -z $key ]] && continue
            
            # Remove quotes if present
            value=$(echo "$value" | sed 's/^"//;s/"$//')
            
            echo "   Setting: $key"
            az webapp config appsettings set \
                --name $APP_NAME \
                --resource-group $RESOURCE_GROUP \
                --settings "$key=$value" > /dev/null
        done < .env
    else
        echo "   ⚠️  No .env file found. Please set environment variables manually in Azure Portal."
    fi
    
    # Set additional required settings
    az webapp config appsettings set \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --settings \
            "WEBSITES_PORT=8000" \
            "SCM_DO_BUILD_DURING_DEPLOYMENT=true" \
            "PYTHON_ENABLE_GUNICORN_MULTIWORKERS=true" > /dev/null
    
    echo "✅ App settings configured!"
}

# Function to deploy application
deploy_app() {
    echo "🚀 Deploying application..."
    
    # Deploy ZIP package
    az webapp deployment source config-zip \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --src healrag-deployment.zip
    
    echo "✅ Application deployed!"
}

# Function to configure startup command
configure_startup() {
    echo "🔧 Configuring startup command..."
    
    az webapp config set \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --startup-file "startup.sh"
    
    echo "✅ Startup command configured!"
}

# Function to show deployment info
show_deployment_info() {
    echo ""
    echo "🎉 Deployment Successful!"
    echo "========================"
    echo "🌐 App URL: https://$APP_NAME.azurewebsites.net"
    echo "🏥 Health Check: https://$APP_NAME.azurewebsites.net/health"
    echo "📚 API Docs: https://$APP_NAME.azurewebsites.net/docs"
    echo "🔐 Login: https://$APP_NAME.azurewebsites.net/auth/login"
    echo ""
    echo "📊 Monitor deployment:"
    echo "   az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
    echo ""
    echo "🔧 Manage in Azure Portal:"
    echo "   https://portal.azure.com/#@/resource/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$APP_NAME"
}

# Function to test deployment
test_deployment() {
    echo "🧪 Testing deployment..."
    
    local url="https://$APP_NAME.azurewebsites.net"
    echo "   Waiting for app to start..."
    sleep 30
    
    # Test health endpoint
    if curl -f "$url/health/simple" > /dev/null 2>&1; then
        echo "✅ Health check passed!"
    else
        echo "⚠️  Health check not responding yet. This is normal for first deployment."
        echo "   Check logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
    fi
}

# Function to clean up temporary files
cleanup() {
    echo "🧹 Cleaning up..."
    rm -rf deploy_temp
    rm -f healrag-deployment.zip
    echo "✅ Cleanup complete!"
}

# Main execution
main() {
    echo "Starting Python direct deployment process..."
    echo ""
    
    validate_azure_cli
    create_deployment_package
    ensure_resource_group
    create_app_service_plan
    create_webapp
    configure_app_settings
    deploy_app
    configure_startup
    show_deployment_info
    test_deployment
    cleanup
    
    echo ""
    echo "🎊 Deployment process complete!"
    echo "Visit https://$APP_NAME.azurewebsites.net to see your application!"
}

# Command line interface
case "$1" in
    "deploy")
        # Parameters already handled above
        main
        ;;
    "test")
        test_deployment
        ;;
    "logs")
        az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP
        ;;
    "clean")
        cleanup
        ;;
    "")
        # No command given, default to deploy
        main
        ;;
    *)
        echo "HEALRAG Python Direct Deployment"
        echo "Usage: $0 [command] [app-name] [resource-group] [location]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Full deployment process (default)"
        echo "  test      - Test existing deployment"
        echo "  logs      - View application logs"
        echo "  clean     - Clean up temporary files"
        echo ""
        echo "Examples:"
        echo "  $0 deploy my-healrag-app my-resource-group westus2"
        echo "  $0 logs my-healrag-app my-resource-group"
        echo "  $0 test my-healrag-app my-resource-group"
        echo ""
        exit 1
        ;;
esac 