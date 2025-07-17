#!/bin/bash

# HEALRAG Docker Deployment Script
# This script builds and deploys the HEALRAG container locally or to Azure

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
IMAGE_NAME="healrag"
TAG="latest"
CONTAINER_NAME="healrag-container"
HOST_PORT="8000"
CONTAINER_PORT="8000"

echo "🐳 HEALRAG Docker Deployment Script"
echo "=================================="

# Function to clean up existing containers
cleanup() {
    echo "🧹 Cleaning up existing containers..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Function to build the image
build_image() {
    echo "🏗️  Building Docker image..."
    docker build -t $IMAGE_NAME:$TAG .
    echo "✅ Image built successfully!"
}

# Function to run the container
run_container() {
    echo "🚀 Starting container..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $HOST_PORT:$CONTAINER_PORT \
        --env-file .env \
        $IMAGE_NAME:$TAG
    
    echo "✅ Container started successfully!"
    echo "📡 Application running at: http://localhost:$HOST_PORT"
    echo "🏥 Health check: http://localhost:$HOST_PORT/health"
}

# Function to show logs
show_logs() {
    echo "📋 Container logs:"
    docker logs $CONTAINER_NAME
}

# Function to test the deployment
test_deployment() {
    echo "🧪 Testing deployment..."
    sleep 5  # Wait for the app to start
    
    # Test health endpoint
    if curl -f http://localhost:$HOST_PORT/health/simple > /dev/null 2>&1; then
        echo "✅ Health check passed!"
    else
        echo "❌ Health check failed!"
        show_logs
        exit 1
    fi
}

# Function to validate Azure environment variables
validate_azure_env() {
    echo "🔍 Validating Azure environment variables..."
    
    if [ -z "$AZURE_CONTAINER_REGISTRY" ]; then
        echo "❌ AZURE_CONTAINER_REGISTRY not found in .env"
        exit 1
    fi
    
    if [ -z "$AZURE_CONTAINER_REGISTRY_USERNAME" ]; then
        echo "❌ AZURE_CONTAINER_REGISTRY_USERNAME not found in .env"
        exit 1
    fi
    
    if [ -z "$AZURE_CONTAINER_REGISTRY_PASSWORD" ]; then
        echo "❌ AZURE_CONTAINER_REGISTRY_PASSWORD not found in .env"
        exit 1
    fi
    
    echo "✅ Azure environment variables validated!"
    echo "📍 Registry: $AZURE_CONTAINER_REGISTRY"
    echo "👤 Username: $AZURE_CONTAINER_REGISTRY_USERNAME"
}

# Function to login to Azure Container Registry
login_azure() {
    echo "🔐 Logging into Azure Container Registry..."
    echo $AZURE_CONTAINER_REGISTRY_PASSWORD | docker login $AZURE_CONTAINER_REGISTRY \
        --username $AZURE_CONTAINER_REGISTRY_USERNAME \
        --password-stdin
    echo "✅ Successfully logged into ACR!"
}

# Function to push to Azure Container Registry
push_azure() {
    REMOTE_IMAGE="$AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:$TAG"
    REMOTE_IMAGE_WITH_SHA="$AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:$(date +%Y%m%d-%H%M%S)"
    
    echo "🏷️  Tagging image for Azure..."
    docker tag $IMAGE_NAME:$TAG $REMOTE_IMAGE
    docker tag $IMAGE_NAME:$TAG $REMOTE_IMAGE_WITH_SHA
    
    echo "📤 Pushing image to Azure Container Registry..."
    echo "   Pushing: $REMOTE_IMAGE"
    docker push $REMOTE_IMAGE
    
    echo "   Pushing: $REMOTE_IMAGE_WITH_SHA"
    docker push $REMOTE_IMAGE_WITH_SHA
    
    echo "✅ Images pushed successfully!"
    echo "📍 Latest image: $REMOTE_IMAGE"
    echo "📍 Timestamped image: $REMOTE_IMAGE_WITH_SHA"
}

# Function to show Azure deployment info
show_azure_info() {
    echo ""
    echo "🎉 Azure Deployment Successful!"
    echo "================================"
    echo "📦 Image: $AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:$TAG"
    echo "🌐 Registry: $AZURE_CONTAINER_REGISTRY"
    echo ""
    echo "🚀 Next steps for Azure App Service deployment:"
    echo "1. Go to Azure Portal > App Services"
    echo "2. Create or select your App Service"
    echo "3. Go to Deployment Center"
    echo "4. Select Container Registry"
    echo "5. Use image: $AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:$TAG"
}

# Function to build AMD64 image for Azure
build_azure_image() {
    echo "🏗️  Building AMD64 Docker image for Azure..."
    docker buildx build --platform linux/amd64 -t $IMAGE_NAME:amd64 .
    echo "✅ AMD64 image built successfully!"
}

# Function to push AMD64 image to Azure
push_azure_amd64() {
    REMOTE_IMAGE_AMD64="$AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:latest-amd64"
    REMOTE_IMAGE_WITH_SHA="$AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:$(date +%Y%m%d-%H%M%S)-amd64"
    
    echo "🏷️  Tagging AMD64 image for Azure..."
    docker tag $IMAGE_NAME:amd64 $REMOTE_IMAGE_AMD64
    docker tag $IMAGE_NAME:amd64 $REMOTE_IMAGE_WITH_SHA
    
    echo "📤 Pushing AMD64 image to Azure Container Registry..."
    echo "   Pushing: $REMOTE_IMAGE_AMD64"
    docker push $REMOTE_IMAGE_AMD64
    
    echo "   Pushing: $REMOTE_IMAGE_WITH_SHA"
    docker push $REMOTE_IMAGE_WITH_SHA
    
    echo "✅ AMD64 images pushed successfully!"
    echo "📍 Latest AMD64 image: $REMOTE_IMAGE_AMD64"
    echo "📍 Timestamped AMD64 image: $REMOTE_IMAGE_WITH_SHA"
}

# Function to validate Azure CLI and resources
validate_azure_cli() {
    echo "🔍 Validating Azure CLI and resources..."
    
    # Check if Azure CLI is logged in
    if ! az account show > /dev/null 2>&1; then
        echo "❌ Azure CLI not logged in. Please run 'az login' first."
        exit 1
    fi
    
    echo "✅ Azure CLI logged in successfully!"
}

# Function to create or update Azure App Service
deploy_webapp() {
    local webapp_name="${1:-healrag-security}"
    local resource_group="${2:-medical}"
    local location="${3:-eastus}"
    local plan_name="${4:-healrag-plan}"
    local sku="${5:-P3v3}"
    
    echo "🚀 Deploying to Azure App Service..."
    echo "   Web App: $webapp_name"
    echo "   Resource Group: $resource_group"
    echo "   Location: $location"
    echo "   Plan: $plan_name ($sku)"
    
    # Check if app service plan exists
    if ! az appservice plan show --name $plan_name --resource-group $resource_group > /dev/null 2>&1; then
        echo "📋 Creating App Service Plan..."
        az appservice plan create \
            --name $plan_name \
            --resource-group $resource_group \
            --location $location \
            --is-linux \
            --sku $sku
        echo "✅ App Service Plan created!"
    else
        echo "✅ App Service Plan already exists!"
    fi
    
    # Check if web app exists
    if ! az webapp show --name $webapp_name --resource-group $resource_group > /dev/null 2>&1; then
        echo "🌐 Creating Web App..."
        az webapp create \
            --name $webapp_name \
            --resource-group $resource_group \
            --plan $plan_name \
            --deployment-container-image-name $AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:latest-amd64
        echo "✅ Web App created!"
    else
        echo "✅ Web App already exists!"
    fi
    
    # Configure container settings
    echo "🔧 Configuring container settings..."
    az webapp config container set \
        --name $webapp_name \
        --resource-group $resource_group \
        --container-image-name $AZURE_CONTAINER_REGISTRY/$IMAGE_NAME:latest-amd64 \
        --container-registry-url https://$AZURE_CONTAINER_REGISTRY \
        --container-registry-user $AZURE_CONTAINER_REGISTRY_USERNAME \
        --container-registry-password $AZURE_CONTAINER_REGISTRY_PASSWORD
    
    # Set startup command
    echo "⚙️  Setting startup command..."
    az webapp config set \
        --name $webapp_name \
        --resource-group $resource_group \
        --startup-file "gunicorn main:app --bind 0.0.0.0:8000 --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --keep-alive 2"
    
    # Configure environment variables from .env file
    echo "🔧 Setting environment variables..."
    configure_webapp_env $webapp_name $resource_group
    
    # Enable continuous deployment
    echo "🔄 Enabling continuous deployment..."
    enable_continuous_deployment $webapp_name $resource_group
    
    # Restart the web app
    echo "🔄 Restarting web app..."
    az webapp restart --name $webapp_name --resource-group $resource_group
    
    # Show deployment info
    show_webapp_info $webapp_name $resource_group
}

# Function to configure webapp environment variables
configure_webapp_env() {
    local webapp_name=$1
    local resource_group=$2
    
    echo "📝 Configuring environment variables..."
    
    # Basic app settings
    az webapp config appsettings set \
        --name $webapp_name \
        --resource-group $resource_group \
        --settings \
            PORT="8000" \
            WEBSITES_PORT="8000" \
            WEBSITES_CONTAINER_START_TIME_LIMIT="1800"
    
    # Azure Storage settings
    if [ ! -z "$AZURE_STORAGE_CONNECTION_STRING" ] && [ ! -z "$AZURE_CONTAINER_NAME" ]; then
        az webapp config appsettings set \
            --name $webapp_name \
            --resource-group $resource_group \
            --settings \
                AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
                AZURE_CONTAINER_NAME="$AZURE_CONTAINER_NAME"
    fi
    
    # Azure OpenAI settings
    if [ ! -z "$AZURE_OPENAI_ENDPOINT" ] && [ ! -z "$AZURE_OPENAI_KEY" ]; then
        az webapp config appsettings set \
            --name $webapp_name \
            --resource-group $resource_group \
            --settings \
                AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
                AZURE_OPENAI_KEY="$AZURE_OPENAI_KEY" \
                AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
                AZURE_TEXT_EMBEDDING_MODEL="$AZURE_TEXT_EMBEDDING_MODEL"
    fi
    
    # Azure Search settings
    if [ ! -z "$AZURE_SEARCH_ENDPOINT" ] && [ ! -z "$AZURE_SEARCH_KEY" ]; then
        az webapp config appsettings set \
            --name $webapp_name \
            --resource-group $resource_group \
            --settings \
                AZURE_SEARCH_ENDPOINT="$AZURE_SEARCH_ENDPOINT" \
                AZURE_SEARCH_KEY="$AZURE_SEARCH_KEY" \
                AZURE_SEARCH_INDEX_NAME="$AZURE_SEARCH_INDEX_NAME"
    fi
    
    # Additional configuration settings
    if [ ! -z "$CHUNK_SIZE" ]; then
        az webapp config appsettings set \
            --name $webapp_name \
            --resource-group $resource_group \
            --settings \
                CHUNK_SIZE="$CHUNK_SIZE" \
                CHUNK_OVERLAP="$CHUNK_OVERLAP" \
                AZURE_SEARCH_INDEXER_NAME="$AZURE_SEARCH_INDEXER_NAME" \
                AZURE_SEARCH_DATASOURCE_NAME="$AZURE_SEARCH_DATASOURCE_NAME" \
                AZURE_SEARCH_SKILLSET_NAME="$AZURE_SEARCH_SKILLSET_NAME" \
                INDEXER_SCHEDULE_MINUTES="$INDEXER_SCHEDULE_MINUTES" \
                VECTOR_PROFILE_SEARCH="$VECTOR_PROFILE_SEARCH"
    fi
    
    # Azure AD settings
    if [ ! -z "$AZURE_AD_TENANT_ID" ] && [ ! -z "$AZURE_AD_CLIENT_ID" ]; then
        # Update redirect URI to match the new web app name
        local redirect_uri="https://nttcodegenerator.azurewebsites.net/auth/callback"
        az webapp config appsettings set \
            --name $webapp_name \
            --resource-group $resource_group \
            --settings \
                AZURE_AD_TENANT_ID="$AZURE_AD_TENANT_ID" \
                AZURE_AD_CLIENT_ID="$AZURE_AD_CLIENT_ID" \
                AZURE_AD_CLIENT_SECRET="$AZURE_AD_CLIENT_SECRET" \
                AZURE_AD_REDIRECT_URI="$redirect_uri"
    fi
    echo "✅ Environment variables configured!"
}

# Function to enable continuous deployment
enable_continuous_deployment() {
    local webapp_name=$1
    local resource_group=$2
    
    echo "🔄 Enabling continuous deployment..."
    
    # Enable container continuous deployment
    az webapp deployment container config \
        --name $webapp_name \
        --resource-group $resource_group \
        --enable-cd true
    
    # Get the webhook URL for continuous deployment
    WEBHOOK_URL=$(az webapp deployment container show-cd-url \
        --name $webapp_name \
        --resource-group $resource_group \
        --query "CI_CD_URL" \
        --output tsv)
    
    echo "✅ Continuous deployment enabled!"
    echo "🔗 Webhook URL: $WEBHOOK_URL"
    echo "💡 Configure this webhook in your Azure Container Registry to enable automatic deployments"
}

# Function to show webapp deployment info
show_webapp_info() {
    local webapp_name=$1
    local resource_group=$2
    
    # Get webapp URL
    WEBAPP_URL=$(az webapp show \
        --name $webapp_name \
        --resource-group $resource_group \
        --query "defaultHostName" \
        --output tsv)
    
    echo ""
    echo "🎉 Azure Web App Deployment Successful!"
    echo "========================================"
    echo "🌐 Web App URL: https://$WEBAPP_URL"
    echo "🏥 Health Check: https://$WEBAPP_URL/health/simple"
    echo "📊 Full Health: https://$WEBAPP_URL/health"
    echo "📋 Management: https://portal.azure.com"
    echo ""
    echo "🔄 Continuous Deployment: ENABLED"
    echo "📝 When you push new images to ACR, the webapp will auto-update"
    echo ""
    echo "🧪 Test your deployment:"
    echo "   curl https://$WEBAPP_URL/health/simple"
}

# Main execution
case "$1" in
    "build")
        build_image
        ;;
    "run")
        cleanup
        run_container
        ;;
    "test")
        test_deployment
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        cleanup
        echo "✅ Container stopped and removed!"
        ;;
    "deploy")
        cleanup
        build_image
        run_container
        test_deployment
        echo "🎉 Local deployment successful!"
        ;;
    "azure-validate")
        validate_azure_env
        ;;
    "azure-login")
        validate_azure_env
        login_azure
        ;;
    "azure-push")
        validate_azure_env
        login_azure
        push_azure
        ;;
    "azure-deploy")
        validate_azure_env
        build_azure_image
        login_azure
        push_azure_amd64
        show_azure_info
        ;;
    "azure-webapp")
        validate_azure_env
        validate_azure_cli
        build_azure_image
        login_azure
        push_azure_amd64
        deploy_webapp
        ;;
    "azure-webapp-only")
        validate_azure_env
        validate_azure_cli
        deploy_webapp "$2" "$3" "$4" "$5" "$6"
        ;;
    "azure-update")
        validate_azure_env
        validate_azure_cli
        build_azure_image
        login_azure
        push_azure_amd64
        ;;
    *)
        echo "Usage: $0 {build|run|test|logs|stop|deploy|azure-validate|azure-login|azure-push|azure-deploy|azure-webapp|azure-webapp-only|azure-update}"
        echo ""
        echo "Local Commands:"
        echo "  build        - Build the Docker image"
        echo "  run          - Run the container locally"
        echo "  test         - Test the local deployment"
        echo "  logs         - Show container logs"
        echo "  stop         - Stop and remove container"
        echo "  deploy       - Full local deployment (build + run + test)"
        echo ""
        echo "Azure Registry Commands:"
        echo "  azure-validate - Check Azure environment variables"
        echo "  azure-login    - Login to Azure Container Registry"
        echo "  azure-push     - Push image to Azure Container Registry"
        echo "  azure-deploy   - Build AMD64 + push to ACR (no webapp)"
        echo "  azure-update   - Build + push new image version to ACR"
        echo ""
        echo "Azure Web App Commands:"
        echo "  azure-webapp      - Full deployment (ACR + Web App + Continuous Deployment)"
        echo "  azure-webapp-only - Deploy/update existing webapp only"
        echo "                     Usage: $0 azure-webapp-only [webapp-name] [resource-group] [location] [plan-name] [sku]"
        echo "                     Example: $0 azure-webapp-only healrag-security medical eastus healrag-plan P3v3"
        echo ""
        echo "Required Azure environment variables in .env:"
        echo "  AZURE_CONTAINER_REGISTRY"
        echo "  AZURE_CONTAINER_REGISTRY_USERNAME"
        echo "  AZURE_CONTAINER_REGISTRY_PASSWORD"
        echo ""
        echo "Optional Azure environment variables (auto-configured if present):"
        echo "  AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME"
        echo "  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT"
        echo "  AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME"
        echo "  CHUNK_SIZE, CHUNK_OVERLAP, etc."
        exit 1
        ;;
esac 