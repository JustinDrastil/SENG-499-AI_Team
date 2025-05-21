import subprocess
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import textwrap

def scrape_website(url: str, max_length: int = 4000) -> str:
    """Scrape and clean text from a webpage with enhanced content extraction"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'img']):
            element.decompose()
        
        # Get clean text with paragraph preservation
        text = '\n\n'.join([
            p.get_text().strip() 
            for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'section'])
            if p.get_text().strip()
        ])
        
        return text[:max_length]
    
    except Exception as e:
        print(f"❌ Scraping failed: {str(e)}")
        return ""

def create_enhanced_model(modelfile_path: str, model_name: str, scraped_content: str = "", target_url: str = "") -> bool:
    """Create an Ollama model with optional scraped content integration"""
    try:
        # Read the base Modelfile
        modelfile = Path(modelfile_path)
        if not modelfile.exists():
            raise FileNotFoundError(f"Modelfile not found at {modelfile_path}")
        
        base_content = modelfile.read_text(encoding='utf-8')
        
        # Enhance with scraped content if provided
        if scraped_content:
            # Extract the original SYSTEM content
            system_start = base_content.find('SYSTEM """') + len('SYSTEM """')
            system_end = base_content.find('"""', system_start)
            original_system = base_content[system_start:system_end]
            
            # Create enhanced content
            enhanced_content = f"""{base_content[:system_start]}
{original_system}

# Documentation Reference: {target_url}
# Scraped Content:
{textwrap.indent(scraped_content, '# ')}
{base_content[system_end:]}
"""
            temp_modelfile = Path("temp_enhanced_modelfile.txt")
            temp_modelfile.write_text(enhanced_content, encoding='utf-8')
            modelfile_to_use = str(temp_modelfile)
        else:
            modelfile_to_use = modelfile_path
        
        # Create the model
        create_cmd = ["ollama", "create", model_name, "-f", modelfile_to_use]
        result = subprocess.run(create_cmd, check=True, capture_output=True, text=True)
        
        print(f"✅ Successfully created model '{model_name}'")
        if scraped_content:
            print("📚 Integrated web content into the model")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create model: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    finally:
        # Clean up temporary file if it exists
        if 'temp_modelfile' in locals() and Path("temp_enhanced_modelfile.txt").exists():
            Path("temp_enhanced_modelfile.txt").unlink()

if __name__ == "__main__":
    # Configuration
    MODELFILE_PATH = "api_query_modelfile.txt"  # Your base Modelfile
    MODEL_NAME = "api-query-enhanced"          # New model name
    TARGET_URL = "https://wiki.oceannetworks.ca/spaces/O2A/pages/48693360/scalardata+service" # Website to scrape
    
    print(f"🌐 Scraping content from {TARGET_URL}...")
    scraped_content = scrape_website(TARGET_URL)
    
    print(f"\n🛠️ Creating '{MODEL_NAME}' with integrated knowledge...")
    if create_enhanced_model(MODELFILE_PATH, MODEL_NAME, scraped_content, TARGET_URL):
        print(f"\n🎉 Done! Use your enhanced model with:")
        print(f"ollama run {MODEL_NAME}")
        print("\nExample query: 'What API parameters would you suggest based on the scraped content?'")
    else:
        print("\n❌ Model creation failed")
        exit(1)