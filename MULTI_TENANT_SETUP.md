# Multi-Tenant Azure AD Authentication Setup

## Problem
You're getting the error `AADSTS50020: User account '341797@NTTDATA.COM' from identity provider 'https://sts.windows.net/65e4e06f-f263-4c1f-becb-90deb8c2d9ff/' does not exist in tenant 'mirakalous.com'` because your application is configured for single-tenant authentication but you need multi-tenant support.

## Solution Overview
To enable multi-tenant authentication, you need to make changes in two places:
1. **Azure Portal**: Configure your App Registration for multi-tenant
2. **Application Code**: Update the authentication logic to handle multi-tenant tokens

## Step 1: Azure Portal Configuration

### 1.1 Update App Registration
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** → **App registrations**
3. Find your app registration (ID: `988232d2-2ffd-43f8-9824-e86ca04a1b74`)
4. Click on **Authentication** in the left menu
5. Under **Supported account types**, change from:
   - ❌ **Accounts in this organizational directory only** (single tenant)
   - ✅ **Accounts in any organizational directory** (multi-tenant)
6. Click **Save**

### 1.2 Update Redirect URIs
Make sure your redirect URI is configured correctly:
- **Redirect URI**: `https://nttcodegenerator.azurewebsites.net/auth/callback`
- **Front-channel logout URL**: `https://nttcodegenerator.azurewebsites.net/`

### 1.3 API Permissions
Ensure your app has the necessary API permissions:
- **Microsoft Graph** → **User.Read** (delegated)
- **Microsoft Graph** → **openid** (delegated)
- **Microsoft Graph** → **profile** (delegated)

## Step 2: Application Code Changes

### 2.1 Updated Configuration
The following changes have been made to `main.py`:

```python
# Azure AD Configuration
AZURE_AD_CONFIG = {
    "tenant_id": os.getenv("AZURE_AD_TENANT_ID"),
    "client_id": os.getenv("AZURE_AD_CLIENT_ID"),
    "client_secret": os.getenv("AZURE_AD_CLIENT_SECRET"),
    "redirect_uri": os.getenv("AZURE_AD_REDIRECT_URI", "https://nttcodegenerator.azurewebsites.net/auth/callback"),
    "authority": "https://login.microsoftonline.com/common",  # Use 'common' for multi-tenant
    "scope": ["openid", "profile", "User.Read"]
}
```

### 2.2 Updated Token Verification
The token verification logic has been updated to:
- Accept tokens from any Azure AD tenant
- Use tenant-specific JWKS for validation
- Log tenant information for debugging

### 2.3 Key Changes Made

1. **Authority URL**: Changed from specific tenant to `common`
2. **JWKS Caching**: Updated to cache JWKS per tenant
3. **Issuer Validation**: Modified to accept any valid Azure AD tenant
4. **Token Verification**: Enhanced for multi-tenant scenarios

## Step 3: Environment Variables

Ensure your `.env` file has the correct configuration:

```bash
# Azure AD Authentication (Multi-Tenant)
AZURE_AD_TENANT_ID=your-home-tenant-id  # Your app's home tenant
AZURE_AD_CLIENT_ID=988232d2-2ffd-43f8-9824-e86ca04a1b74
AZURE_AD_CLIENT_SECRET=your-client-secret
AZURE_AD_REDIRECT_URI=https://nttcodegenerator.azurewebsites.net/auth/callback
```

## Step 4: Testing Multi-Tenant Authentication

### 4.1 Test with Different Tenants
1. **Deploy the updated code**
2. **Test with users from different tenants**:
   - Users from your home tenant (`mirakalous.com`)
   - Users from external tenants (like `NTTDATA.COM`)

### 4.2 Debug Endpoints
Use these endpoints to debug authentication:

```bash
# Test token validation
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://nttcodegenerator.azurewebsites.net/debug/token

# Test authentication
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://nttcodegenerator.azurewebsites.net/auth/me
```

## Step 5: Admin Consent (Important!)

### 5.1 For External Tenants
When users from external tenants try to access your app for the first time, their admin may need to grant consent:

1. **User Consent**: Individual users can consent to basic permissions
2. **Admin Consent**: Required for sensitive permissions or organizational policies

### 5.2 Admin Consent URL
Provide this URL to external tenant admins:
```
https://login.microsoftonline.com/common/adminconsent?client_id=988232d2-2ffd-43f8-9824-e86ca04a1b74&redirect_uri=https://nttcodegenerator.azurewebsites.net/auth/callback
```

## Step 6: Deployment

### 6.1 Update Environment Variables
```bash
# Update your Azure App Service environment variables
az webapp config appsettings set \
  --name NTTCodeGenerator \
  --resource-group your-resource-group \
  --settings \
    AZURE_AD_TENANT_ID="your-home-tenant-id" \
    AZURE_AD_CLIENT_ID="988232d2-2ffd-43f8-9824-e86ca04a1b74" \
    AZURE_AD_CLIENT_SECRET="your-client-secret" \
    AZURE_AD_REDIRECT_URI="https://nttcodegenerator.azurewebsites.net/auth/callback"
```

### 6.2 Deploy Updated Code
```bash
# Deploy the updated application
./deploy.sh azure-update
```

## Troubleshooting

### Common Issues

1. **AADSTS50020 Error Still Occurs**
   - Ensure App Registration is set to multi-tenant
   - Check that admin consent has been granted
   - Verify redirect URI is correct

2. **Token Validation Fails**
   - Check logs for JWKS retrieval errors
   - Verify client ID matches in token
   - Ensure proper scopes are requested

3. **External Users Can't Access**
   - Admin consent may be required
   - Check tenant's security policies
   - Verify app permissions are sufficient

### Debug Commands

```bash
# Check current app registration settings
az ad app show --id 988232d2-2ffd-43f8-9824-e86ca04a1b74 --query "signInAudience"

# List app permissions
az ad app permission list --id 988232d2-2ffd-43f8-9824-e86ca04a1b74

# Check app service environment variables
az webapp config appsettings list --name NTTCodeGenerator --resource-group your-resource-group
```

## Security Considerations

1. **Token Validation**: Always validate tokens server-side
2. **Scope Limitation**: Request only necessary permissions
3. **Audit Logging**: Monitor authentication attempts
4. **Rate Limiting**: Implement rate limiting for auth endpoints
5. **HTTPS Only**: Ensure all communication uses HTTPS

## Monitoring

### Key Metrics to Monitor
- Authentication success/failure rates
- Token validation errors
- JWKS retrieval performance
- Cross-tenant access patterns

### Log Analysis
Look for these log patterns:
- `Token from tenant: {tenant_id}` - Successful multi-tenant auth
- `Failed to get JWKS for tenant {tenant_id}` - JWKS issues
- `Invalid issuer format` - Token validation problems

## Next Steps

1. **Deploy the updated code**
2. **Test with users from different tenants**
3. **Monitor authentication logs**
4. **Set up admin consent process for external tenants**
5. **Implement additional security measures if needed**

## Support

If you continue to experience issues:
1. Check Azure AD audit logs
2. Review application logs
3. Test with the debug endpoints
4. Verify all configuration changes are deployed 