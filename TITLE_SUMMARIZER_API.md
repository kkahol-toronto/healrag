# Title Summarizer API Documentation

## Overview

The Title Summarizer endpoint generates concise, meaningful titles from text using AI. It's useful for creating document titles, section headers, search-friendly titles, and meeting agenda items.

## Endpoint Details

- **URL**: `POST /title-summarizer`
- **Authentication**: Required (Bearer token)
- **Content-Type**: `application/json`

## Request Format

```json
{
  "text": "The text you want to summarize into a title",
  "max_words": 4  // Optional - defaults to 4 if not provided
}
```

### Parameters

| Parameter | Type | Required | Default | Range | Description |
|-----------|------|----------|---------|-------|-------------|
| `text` | string | ✅ Yes | - | - | The text to summarize into a title |
| `max_words` | integer | ❌ No | 4 | 1-20 | Maximum number of words for the title |

## Response Format

```json
{
  "success": true,
  "summary": "Generated Title Here",
  "word_count": 4,
  "original_text": "The original text you sent",
  "error": null
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request was successful |
| `summary` | string | The generated title |
| `word_count` | integer | Number of words in the generated title |
| `original_text` | string | The original input text |
| `error` | string/null | Error message if request failed |

## Examples

### Basic Request (Default 4 Words)

**Request:**
```bash
curl -X POST "http://localhost:8000/title-summarizer" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "text": "This is a comprehensive guide about cybersecurity policies and procedures that organizations should follow to protect their digital assets and maintain compliance with industry standards."
  }'
```

**Response:**
```json
{
  "success": true,
  "summary": "Cybersecurity Policies Compliance Guide",
  "word_count": 4,
  "original_text": "This is a comprehensive guide about cybersecurity policies and procedures that organizations should follow to protect their digital assets and maintain compliance with industry standards.",
  "error": null
}
```

### Custom Word Limit

**Request:**
```bash
curl -X POST "http://localhost:8000/title-summarizer" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "text": "This is a comprehensive guide about cybersecurity policies and procedures that organizations should follow to protect their digital assets and maintain compliance with industry standards.",
    "max_words": 6
  }'
```

**Response:**
```json
{
  "success": true,
  "summary": "Cybersecurity Policies Guide for Organizational Compliance",
  "word_count": 6,
  "original_text": "This is a comprehensive guide about cybersecurity policies and procedures that organizations should follow to protect their digital assets and maintain compliance with industry standards.",
  "error": null
}
```

### Short Text Example

**Request:**
```bash
curl -X POST "http://localhost:8000/title-summarizer" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "text": "Cybersecurity guide for organizations"
  }'
```

**Response:**
```json
{
  "success": true,
  "summary": "Organizational Cybersecurity Best Practices",
  "word_count": 4,
  "original_text": "Cybersecurity guide for organizations",
  "error": null
}
```

## Configuration

The endpoint reads configuration from environment variables:

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TITLE_SUMMARIZER_MAX_WORDS` | `4` | Default word limit for titles |
| `TITLE_SUMMARIZER_PROMPT` | See below | Customizable AI prompt |

### Default Prompt

```
You are a title generator. Summarize the following text into exactly {max_words} words. Respond with ONLY the title, no explanations or additional text. Text to summarize: {text}
```

### Customizing the Prompt

You can customize the AI prompt by setting the `TITLE_SUMMARIZER_PROMPT` environment variable. The prompt must include:
- `{max_words}` placeholder for the word limit
- `{text}` placeholder for the input text

Example custom prompt:
```bash
export TITLE_SUMMARIZER_PROMPT="Create a {max_words}-word title that captures the main topic. Be concise and professional. Text: {text}"
```

## Error Handling

### Common Errors

**LLM Manager Not Available:**
```json
{
  "success": false,
  "summary": "",
  "word_count": 0,
  "original_text": "Your text here",
  "error": "LLM Manager not available"
}
```

**Invalid Request:**
```json
{
  "success": false,
  "summary": "",
  "word_count": 0,
  "original_text": "",
  "error": "Validation error: text field required"
}
```

### Error Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `false` for errors |
| `summary` | string | Empty string |
| `word_count` | integer | Always `0` |
| `original_text` | string | Original text (if provided) |
| `error` | string | Error message |

## Use Cases

### 1. Document Title Generation
Generate titles for documents based on their content:
```javascript
const title = await generateTitle(documentContent, 5);
// Result: "Comprehensive Security Policy Framework"
```

### 2. Section Headers
Create headers for document sections:
```javascript
const header = await generateTitle(sectionContent, 3);
// Result: "Access Control Policies"
```

### 3. Search-Friendly Titles
Generate titles optimized for search:
```javascript
const searchTitle = await generateTitle(queryText, 4);
// Result: "Cybersecurity Best Practices Guide"
```

### 4. Meeting Agenda Items
Create concise agenda item titles:
```javascript
const agendaItem = await generateTitle(meetingTopic, 2);
// Result: "Security Review"
```

## Frontend Integration

### JavaScript Example

```javascript
async function generateTitle(text, maxWords = 4) {
  try {
    const response = await fetch('/title-summarizer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${getAuthToken()}`
      },
      body: JSON.stringify({
        text: text,
        max_words: maxWords
      })
    });

    const result = await response.json();
    
    if (result.success) {
      return result.summary;
    } else {
      throw new Error(result.error);
    }
  } catch (error) {
    console.error('Title generation failed:', error);
    return null;
  }
}

// Usage
const title = await generateTitle('Your text here', 4);
console.log(title); // "Generated Title Here"
```

### React Hook Example

```javascript
import { useState } from 'react';

function useTitleSummarizer() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateTitle = async (text, maxWords = 4) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/title-summarizer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${getAuthToken()}`
        },
        body: JSON.stringify({ text, max_words: maxWords })
      });

      const result = await response.json();
      
      if (result.success) {
        return result.summary;
      } else {
        throw new Error(result.error);
      }
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { generateTitle, loading, error };
}
```

## Rate Limiting

The title summarizer endpoint is subject to the same rate limiting as other authenticated endpoints. Consider implementing client-side caching for frequently requested titles.

## Performance

- **Typical Response Time**: 200-500ms
- **Token Usage**: ~50 tokens per request
- **Concurrent Requests**: Limited by Azure OpenAI rate limits

## Troubleshooting

### Common Issues

1. **"LLM Manager not available"**
   - Check Azure OpenAI configuration
   - Verify environment variables are set

2. **"Validation error"**
   - Ensure `text` field is provided
   - Check `max_words` is between 1-20

3. **Slow responses**
   - Check Azure OpenAI service status
   - Consider reducing concurrent requests

### Debug Mode

Enable debug logging by setting the log level to DEBUG in your application configuration. 