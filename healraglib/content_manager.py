"""
Content Manager for HEALRAG

Handles content extraction from various file formats, image processing,
and markdown generation with Azure OpenAI integration.
"""

import os
import json
import logging
import tempfile
import base64
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import mimetypes
from datetime import datetime

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    MarkItDown = None
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Optional: Azure OpenAI imports
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

class ContentManager:
    """
    Content Manager for HEALRAG library.
    
    Handles content extraction, image processing, and markdown generation
    with support for Azure OpenAI image analysis.
    """
    
    def __init__(self, 
                 storage_manager,
                 azure_openai_endpoint: Optional[str] = None,
                 azure_openai_key: Optional[str] = None,
                 azure_openai_deployment: Optional[str] = None):
        """
        Initialize the Content Manager.
        
        Args:
            storage_manager: HEALRAG StorageManager instance
            azure_openai_endpoint: Azure OpenAI endpoint
            azure_openai_key: Azure OpenAI API key
            azure_openai_deployment: Azure OpenAI deployment name
        """
        self.storage_manager = storage_manager
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_key = azure_openai_key
        self.azure_openai_deployment = azure_openai_deployment
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI client if credentials provided
        self.openai_client = None
        if all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
            self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client."""
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.azure_openai_endpoint,
                api_key=self.azure_openai_key,
                api_version="2024-02-15-preview"
            )
            self.logger.info("Azure OpenAI client initialized successfully")
        except ImportError:
            self.logger.warning("openai package not installed. Install with: pip install openai")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    
    def extract_content_from_files(self, 
                                 file_list: List[str], 
                                 output_folder: str = "md_files",
                                 image_prompt: str = "Describe this image in detail, including any text, charts, diagrams, or visual elements that would be important for understanding the document content.",
                                 extract_images: bool = True) -> Dict[str, Dict]:
        """
        Extract content from a list of files and generate markdown files.
        
        Args:
            file_list: List of file paths (local or blob names)
            output_folder: Folder name in blob storage for markdown files
            image_prompt: Prompt for image analysis
            extract_images: Whether to extract and process images
            
        Returns:
            Dict: Results of content extraction for each file
        """
        results = {}
        
        for file_path in file_list:
            try:
                self.logger.info(f"Processing file: {file_path}")
                
                # Determine if file is local or blob
                if os.path.exists(file_path):
                    # Local file
                    result = self._process_local_file(
                        file_path, output_folder, image_prompt, extract_images
                    )
                else:
                    # Blob file
                    result = self._process_blob_file(
                        file_path, output_folder, image_prompt, extract_images
                    )
                
                results[file_path] = result
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                results[file_path] = {
                    "success": False,
                    "error": str(e),
                    "markdown_file": None,
                    "images_processed": 0
                }
        
        return results
    
    def _process_local_file(self, 
                           file_path: str, 
                           output_folder: str,
                           image_prompt: str,
                           extract_images: bool,
                           original_filename: Optional[str] = None) -> Dict:
        """Process a local file and generate markdown."""
        try:
            # Extract base content using MarkItDown
            if not MARKITDOWN_AVAILABLE:
                raise ImportError("MarkItDown is not available. Please install it with: pip install markitdown[all]")
            
            # Use MarkItDown directly without Document Intelligence
            md = MarkItDown()
            result = md.convert(file_path)
            
            # Debug logging
            self.logger.info(f"MarkItDown result type: {type(result)}")
            self.logger.info(f"MarkItDown result: {result}")
            
            # Handle case where result might be None or doesn't have text_content
            if result is None:
                raise Exception("MarkItDown conversion returned None")
            
            if hasattr(result, 'text_content'):
                base_content = result.text_content
                self.logger.info(f"Extracted {len(base_content)} characters using text_content")
            elif hasattr(result, 'text'):
                base_content = result.text
                self.logger.info(f"Extracted {len(base_content)} characters using text")
            elif isinstance(result, str):
                base_content = result
                self.logger.info(f"Extracted {len(base_content)} characters using string result")
            else:
                # Try to get content from result object
                base_content = str(result)
                self.logger.info(f"Extracted {len(base_content)} characters using string conversion")
            
        except (ImportError, AttributeError, Exception) as e:
            # Fallback to basic text extraction for supported file types
            self.logger.warning(f"MarkItDown extraction failed: {e}. Using fallback method.")
            try:
                base_content = self._fallback_content_extraction(file_path)
            except Exception as fallback_error:
                self.logger.error(f"Fallback extraction also failed: {fallback_error}")
                base_content = f"[Content extraction failed for {Path(file_path).name}. Error: {str(e)}]"
        
        # Generate markdown content
        markdown_content, images_processed = self._generate_markdown_content(
            file_path, base_content, image_prompt, extract_images
        )
        
        # Use the original filename if provided, otherwise use the file_path
        if original_filename:
            file_name = Path(original_filename).name  # includes original filename with extension
        else:
            file_name = Path(file_path).name  # includes original filename with extension
        markdown_filename = f"{output_folder}/{Path(file_name).stem}.md"
        
        # Upload markdown to blob storage (overwrite if exists)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_file.write(markdown_content)
            temp_file_path = temp_file.name
        
        try:
            # Overwrite is default for upload_file, but make it explicit in comments
            success = self.storage_manager.upload_file(temp_file_path, markdown_filename)
            if success:
                self.logger.info(f"Markdown uploaded (overwritten if existed): {markdown_filename}")
            else:
                raise Exception("Failed to upload markdown file")
        finally:
            os.unlink(temp_file_path)
        
        return {
            "success": True,
            "markdown_file": markdown_filename,
            "images_processed": images_processed,
            "content_length": len(markdown_content)
        }
        
    def _process_blob_file(self, 
                          blob_name: str, 
                          output_folder: str,
                          image_prompt: str,
                          extract_images: bool) -> Dict:
        """Process a blob file and generate markdown."""
        try:
            # Download blob to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(blob_name).suffix) as temp_file:
                temp_file_path = temp_file.name
            
            try:
                # Download the file
                if not self.storage_manager.download_file(blob_name, temp_file_path):
                    raise Exception(f"Failed to download blob: {blob_name}")
                
                # Process the downloaded file, passing the original blob name
                result = self._process_local_file(
                    temp_file_path, output_folder, image_prompt, extract_images, original_filename=blob_name
                )
                
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            self.logger.error(f"Error processing blob file {blob_name}: {e}")
            raise
    
    def _generate_markdown_content(self, 
                                 file_path: str, 
                                 base_content: str,
                                 image_prompt: str,
                                 extract_images: bool) -> Tuple[str, int]:
        """
        Generate markdown content with image processing.
        
        Returns:
            Tuple[str, int]: Markdown content and number of images processed
        """
        file_ext = Path(file_path).suffix.lower()
        images_processed = 0
        
        # Start with base content
        markdown_content = f"# {Path(file_path).stem}\n\n"
        markdown_content += f"**Source:** {file_path}\n"
        markdown_content += f"**Extracted:** {datetime.now().isoformat()}\n\n"
        
        # Add base content
        markdown_content += base_content + "\n\n"
        
        # Extract and process images if requested
        if extract_images and file_ext in ['.pdf', '.docx', '.doc']:
            self.logger.info(f"Attempting to extract images from {file_path}")
            try:
                image_results = self._extract_and_process_images(
                    file_path, image_prompt
                )
                
                if image_results:
                    markdown_content += "## Extracted Images\n\n"
                    
                    for i, image_result in enumerate(image_results, 1):
                        markdown_content += f"### Image {i}\n\n"
                        markdown_content += f"**Page:** {image_result.get('page', 'Unknown')}\n"
                        markdown_content += f"**Section:** {image_result.get('section', 'Unknown')}\n"
                        markdown_content += f"**Image Type:** {image_result.get('image_type', 'Unknown')}\n"
                        markdown_content += f"**File Size:** {image_result.get('file_size', 'Unknown')} bytes\n\n"
                        
                        if image_result.get('description'):
                            markdown_content += f"**Description:** {image_result['description']}\n\n"
                        else:
                            markdown_content += f"**Note:** Image found but no description available (Azure OpenAI not configured)\n\n"
                        
                        if image_result.get('metadata'):
                            markdown_content += "**Metadata:**\n"
                            for key, value in image_result['metadata'].items():
                                markdown_content += f"- {key}: {value}\n"
                            markdown_content += "\n"
                        
                        markdown_content += "---\n\n"
                        images_processed += 1
                else:
                    markdown_content += "**Note:** No images found in document\n\n"
                        
            except Exception as e:
                self.logger.warning(f"Image extraction failed for {file_path}: {e}")
                markdown_content += f"**Note:** Image extraction failed: {str(e)}\n\n"
        
        return markdown_content, images_processed
    
    def _extract_and_process_images(self, 
                                   file_path: str, 
                                   image_prompt: str) -> List[Dict]:
        """
        Extract images from PDF/Word documents and process them.
        
        Returns:
            List[Dict]: List of image processing results
        """
        image_results = []
        file_ext = Path(file_path).suffix.lower()
        
        self.logger.info(f"Starting image extraction for {file_path} (type: {file_ext})")
        
        try:
            if file_ext == '.pdf':
                self.logger.info("Extracting images from PDF")
                image_results = self._extract_images_from_pdf(file_path, image_prompt)
            elif file_ext in ['.docx', '.doc']:
                self.logger.info("Extracting images from Word document")
                image_results = self._extract_images_from_word(file_path, image_prompt)
            else:
                self.logger.info(f"File type {file_ext} not supported for image extraction")
                
        except Exception as e:
            self.logger.error(f"Error extracting images from {file_path}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.logger.info(f"Image extraction completed. Found {len(image_results)} images")
        return image_results
    
    def _extract_images_from_pdf(self, file_path: str, image_prompt: str) -> List[Dict]:
        """Extract images from PDF file."""
        image_results = []
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Save image to temporary file
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                            pix.save(temp_img.name)
                            temp_img_path = temp_img.name
                        
                        try:
                            # Process image with Azure OpenAI
                            image_result = self._process_image_with_openai(
                                temp_img_path, image_prompt, page_num + 1, img_index + 1
                            )
                            
                            if image_result:
                                image_results.append(image_result)
                                
                        finally:
                            # Clean up temporary image
                            if os.path.exists(temp_img_path):
                                os.unlink(temp_img_path)
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing image {img_index} on page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except ImportError:
            self.logger.warning("PyMuPDF not installed. Install with: pip install PyMuPDF")
        except Exception as e:
            self.logger.error(f"Error extracting images from PDF {file_path}: {e}")
        
        return image_results
    
    def _extract_images_from_word(self, file_path: str, image_prompt: str) -> List[Dict]:
        """Extract images from Word document."""
        image_results = []
        
        try:
            from docx import Document
            
            doc = Document(file_path)
            
            for para_index, paragraph in enumerate(doc.paragraphs):
                for run in paragraph.runs:
                    for drawing in run._element.findall('.//w:drawing', {'w': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}):
                        try:
                            image_data = self._extract_image_from_drawing(drawing)
                            if image_data:
                                # Save image to temporary file
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                                    temp_img.write(image_data)
                                    temp_img_path = temp_img.name
                                
                                try:
                                    # Process image with Azure OpenAI
                                    image_result = self._process_image_with_openai(
                                        temp_img_path, image_prompt, 1, len(image_results) + 1
                                    )
                                    
                                    if image_result:
                                        image_results.append(image_result)
                                        
                                finally:
                                    # Clean up temporary image
                                    if os.path.exists(temp_img_path):
                                        os.unlink(temp_img_path)
                                        
                        except Exception as e:
                            self.logger.warning(f"Error processing drawing in paragraph {para_index}: {e}")
                            continue
            
        except ImportError:
            self.logger.warning("python-docx not installed. Install with: pip install python-docx")
        except Exception as e:
            self.logger.error(f"Error extracting images from Word document {file_path}: {e}")
        
        return image_results
    
    def _extract_image_from_drawing(self, drawing_element) -> Optional[bytes]:
        """Extract image data from a drawing element in Word document."""
        try:
            # Find the image element
            blip = drawing_element.find('.//a:blip', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
            if blip is not None:
                embed = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if embed:
                    # This would require accessing the relationship to get the actual image data
                    # For now, return None as this is complex to implement
                    return None
        except Exception as e:
            self.logger.warning(f"Error extracting image from drawing: {e}")
        
        return None
    
    def _process_image_with_openai(self, 
                                  image_path: str, 
                                  prompt: str,
                                  page_num: int,
                                  image_index: int) -> Optional[Dict]:
        """
        Process image with Azure OpenAI Vision API.
        
        Returns:
            Dict: Image processing result with metadata and description
        """
        if not self.openai_client:
            self.logger.warning("Azure OpenAI not configured. Skipping image analysis.")
            return {
                "page": page_num,
                "image_index": image_index,
                "image_type": "Unknown",
                "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                "description": None,
                "metadata": {
                    "note": "Azure OpenAI not configured for image analysis"
                }
            }
        
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Convert to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Call Azure OpenAI Vision API
            response = self.openai_client.chat.completions.create(
                model=self.azure_openai_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            description = response.choices[0].message.content
            
            # Get image metadata
            file_size = os.path.getsize(image_path)
            image_type = "PNG"  # Default, could be enhanced to detect actual type
            
            return {
                "page": page_num,
                "image_index": image_index,
                "image_type": image_type,
                "file_size": file_size,
                "description": description,
                "metadata": {
                    "model": self.azure_openai_deployment,
                    "prompt": prompt,
                    "response_tokens": len(description.split())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image with Azure OpenAI: {e}")
            return {
                "page": page_num,
                "image_index": image_index,
                "image_type": "Unknown",
                "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                "description": None,
                "metadata": {
                    "error": str(e)
                }
            }
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types for content extraction."""
        return ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.rtf']
    
    def get_image_extraction_support(self) -> List[str]:
        """Get list of file types that support image extraction."""
        return ['.pdf', '.docx', '.doc']
    
    def get_source_files_from_container(self, exclude_folders: List[str] = None) -> List[str]:
        """
        Get list of source files from container, excluding specified folders and unsupported file types.
        
        Args:
            exclude_folders: List of folder names to exclude (default: ['md_files'])
            
        Returns:
            List of source file paths ready for processing
        """
        if exclude_folders is None:
            exclude_folders = ['md_files']
        
        # Get all files from container
        all_files = self.storage_manager.get_file_list(as_json=False)
        
        # Filter out excluded folders and unsupported file types
        supported_types = self.get_supported_file_types()
        source_files = []
        
        for file_path in all_files:
            # Skip files in excluded folders
            if any(file_path.startswith(folder + '/') for folder in exclude_folders):
                continue
            
            # Only include supported file types
            if Path(file_path).suffix.lower() in supported_types:
                source_files.append(file_path)
        
        return source_files
    
    def _fallback_content_extraction(self, file_path: str) -> str:
        """Fallback content extraction for basic file types."""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_ext == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"[Fallback extraction not supported for {file_ext} files]"
        except Exception as e:
            return f"[Error reading file: {str(e)}]"
