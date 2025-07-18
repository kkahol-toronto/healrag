# Anonymous Access Feature

## Overview

The NTTCodeGenerator application now supports **anonymous access** for basic features, allowing users to test the system without requiring Azure AD authentication. This is perfect for demos, testing, and users who want to try the system before signing up.

## 🎯 Anonymous Access Features

### Available Endpoints

1. **RAG Queries** (`/rag/query/anonymous`)
   - Ask questions and get AI-powered answers
   - Limited to 300 tokens per response
   - Limited to 2 source documents
   - No session persistence

2. **Document Search** (`/search/documents/anonymous`)
   - Search through uploaded documents
   - Limited to 3 search results
   - Basic search functionality

3. **Anonymous User Info** (`/auth/anonymous`)
   - Get anonymous user information
   - Useful for testing

4. **Anonymous Test** (`/anonymous/test`)
   - Test endpoint to verify anonymous access
   - Shows available features and limitations

## 🚀 How to Use Anonymous Access

### 1. Test Anonymous Access

```bash
# Test if anonymous access is working
curl https://nttcodegenerator.azurewebsites.net/anonymous/test
```

**Expected Response:**
```json
{
    "message": "🎉 Anonymous access is working!",
    "user_info": {
        "user_id": "anonymous",
        "email": "anonymous@example.com",
        "name": "Anonymous User",
        "is_anonymous": true
    },
    "available_endpoints": {
        "rag_query": "/rag/query/anonymous",
        "search_documents": "/search/documents/anonymous",
        "anonymous_user": "/auth/anonymous"
    },
    "limitations": {
        "max_tokens": 300,
        "max_sources": 2,
        "max_search_results": 3,
        "no_session_persistence": true
    }
}
```

### 2. Make RAG Queries (Anonymous)

```bash
# Ask a question without authentication
curl -X POST https://nttcodegenerator.azurewebsites.net/rag/query/anonymous \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the security policies?",
    "top_k": 2,
    "max_tokens": 300
  }'
```

**Example Response:**
```json
{
    "message": {
        "content": "Based on the security documents, the main security policies include...",
        "role": "assistant"
    },
    "context": {
        "data_points": [...],
        "followup_questions": [...],
        "thoughts": [...]
    },
    "sources": [
        {
            "title": "Security Policy Document",
            "content": "...",
            "source": "security_policy.pdf",
            "chunk_id": "...",
            "score": 0.85
        }
    ],
    "session_state": "anonymous_1234567890",
    "user_info": {
        "user_id": "anonymous",
        "email": "anonymous@example.com",
        "name": "Anonymous User",
        "is_anonymous": true
    }
}
```

### 3. Search Documents (Anonymous)

```bash
# Search documents without authentication
curl -X POST https://nttcodegenerator.azurewebsites.net/search/documents/anonymous \
  -H "Content-Type: application/json" \
  -d '{
    "query": "security policy",
    "top_k": 3
  }'
```

**Example Response:**
```json
{
    "success": true,
    "results": [
        {
            "content": "Security policy content...",
            "source": "security_policy.pdf",
            "score": 0.92
        }
    ],
    "metadata": {
        "query": "security policy",
        "top_k": 3,
        "results_count": 1,
        "user_info": {
            "user_id": "anonymous",
            "email": "anonymous@example.com",
            "name": "Anonymous User",
            "is_anonymous": true
        }
    }
}
```

## 📋 Limitations for Anonymous Users

### RAG Queries
- **Max Tokens**: 300 (vs 500 for authenticated users)
- **Max Sources**: 2 (vs 3 for authenticated users)
- **Session Persistence**: No (vs Yes for authenticated users)
- **Custom System Prompts**: Limited (adds anonymous user note)

### Document Search
- **Max Results**: 3 (vs 5 for authenticated users)
- **Advanced Features**: Limited

### General Limitations
- **No Session History**: Anonymous users can't access chat history
- **No Training**: Can't start training pipelines
- **No Configuration**: Can't modify system settings
- **No Storage Stats**: Can't view storage statistics

## 🔐 Security Considerations

### What Anonymous Users Can Access
- ✅ Basic RAG queries (limited)
- ✅ Document search (limited)
- ✅ Public health endpoints
- ✅ API documentation

### What Anonymous Users Cannot Access
- ❌ Full RAG capabilities
- ❌ Session management
- ❌ Training pipelines
- ❌ System configuration
- ❌ Storage statistics
- ❌ User-specific data

### Rate Limiting
- Anonymous users are subject to the same rate limiting as authenticated users
- Consider implementing stricter rate limits for anonymous access if needed

## 🎨 Frontend Integration

### Anonymous Access Button
Add an "Anonymous Access" button next to the login button:

```html
<div class="auth-buttons">
    <button onclick="window.location.href='/auth/login'">Login</button>
    <button onclick="enableAnonymousMode()">Anonymous Access</button>
</div>
```

### JavaScript for Anonymous Mode

```javascript
function enableAnonymousMode() {
    // Store anonymous mode preference
    localStorage.setItem('anonymousMode', 'true');
    
    // Update UI for anonymous mode
    document.getElementById('user-info').innerHTML = `
        <div class="anonymous-user">
            <span>👤 Anonymous User</span>
            <small>Limited access mode</small>
        </div>
    `;
    
    // Update API endpoints to use anonymous versions
    window.API_BASE = {
        rag: '/rag/query/anonymous',
        search: '/search/documents/anonymous'
    };
    
    // Show limitations notice
    showNotification('Anonymous mode enabled. Some features are limited.');
}

function checkAnonymousMode() {
    if (localStorage.getItem('anonymousMode') === 'true') {
        enableAnonymousMode();
    }
}
```

## 🧪 Testing Anonymous Access

### Test Scripts

```bash
#!/bin/bash

# Test anonymous access
echo "Testing anonymous access..."

# Test anonymous endpoint
curl -s https://nttcodegenerator.azurewebsites.net/anonymous/test | jq '.'

# Test anonymous RAG query
curl -s -X POST https://nttcodegenerator.azurewebsites.net/rag/query/anonymous \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this system about?", "max_tokens": 200}' | jq '.'

# Test anonymous search
curl -s -X POST https://nttcodegenerator.azurewebsites.net/search/documents/anonymous \
  -H "Content-Type: application/json" \
  -d '{"query": "security", "top_k": 2}' | jq '.'
```

### Expected Behavior

1. **Anonymous Test**: Should return success with user info
2. **RAG Query**: Should work with limitations applied
3. **Search**: Should return limited results
4. **Protected Endpoints**: Should return 401 Unauthorized

## 🔧 Configuration

### Environment Variables

No additional environment variables are required for anonymous access. The feature is enabled by default.

### Disabling Anonymous Access

To disable anonymous access, you can:

1. **Remove anonymous endpoints** from the code
2. **Add authentication checks** to anonymous endpoints
3. **Implement IP-based restrictions**

### Customizing Limitations

You can modify the limitations in `main.py`:

```python
# In rag_query_anonymous function
if current_user.is_anonymous:
    # Adjust these values as needed
    if request.max_tokens > 300:  # Change from 300
        request.max_tokens = 300
    if request.top_k > 2:  # Change from 2
        request.top_k = 2
```

## 📊 Monitoring Anonymous Usage

### Log Patterns

Look for these log patterns:
- `Anonymous user query` - Anonymous RAG queries
- `Token from tenant: anonymous` - Anonymous authentication
- `Anonymous search request` - Anonymous search queries

### Metrics to Track

- Anonymous vs authenticated user ratio
- Anonymous query success/failure rates
- Anonymous user conversion to authenticated users
- Anonymous endpoint usage patterns

## 🚀 Deployment

### Deploy with Anonymous Access

```bash
# Deploy the updated application
./deploy.sh azure-update

# Test anonymous access
curl https://nttcodegenerator.azurewebsites.net/anonymous/test
```

### Verification Checklist

- [ ] Anonymous test endpoint responds correctly
- [ ] Anonymous RAG queries work with limitations
- [ ] Anonymous search works with limitations
- [ ] Protected endpoints still require authentication
- [ ] Rate limiting works for anonymous users
- [ ] Logging captures anonymous user activity

## 🆘 Troubleshooting

### Common Issues

1. **Anonymous endpoints return 401**
   - Check that the endpoints are properly configured
   - Verify the `get_user_or_anonymous` dependency is working

2. **Anonymous users get full access**
   - Check that limitations are properly applied
   - Verify the `is_anonymous` flag is set correctly

3. **Anonymous mode not working in frontend**
   - Check JavaScript console for errors
   - Verify localStorage is working
   - Check API endpoint URLs

### Debug Commands

```bash
# Test anonymous access
curl -v https://nttcodegenerator.azurewebsites.net/anonymous/test

# Check if anonymous endpoints exist
curl -v https://nttcodegenerator.azurewebsites.net/docs

# Test with and without authentication
curl -H "Authorization: Bearer INVALID_TOKEN" \
  https://nttcodegenerator.azurewebsites.net/rag/query/anonymous
```

## 📈 Future Enhancements

### Potential Improvements

1. **Anonymous User Analytics**
   - Track anonymous user behavior
   - Conversion funnel analysis

2. **Progressive Limitations**
   - Time-based limitations
   - Usage-based restrictions

3. **Anonymous User Onboarding**
   - Guided tour for anonymous users
   - Feature comparison table

4. **Anonymous to Authenticated Conversion**
   - Seamless upgrade process
   - Data migration for anonymous sessions

## 📞 Support

For issues with anonymous access:

1. Check the application logs
2. Test with the provided test scripts
3. Verify endpoint configurations
4. Review rate limiting settings

Anonymous access provides a great way for users to try your system before committing to authentication, while maintaining security for sensitive operations. 