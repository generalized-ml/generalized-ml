
import requests
from bs4 import BeautifulSoup

def get_website_text(url):
    """
    Fetches the HTML content of a given URL and extracts all visible text.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        str: All the visible text content from the website, or an error message.
    """
    try:
        # Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all text content
        # .get_text() method extracts all text within a tag and its descendants
        # strip=True removes leading/trailing whitespace from each line
        # separator='\n' adds a newline between extracted text elements
        all_text = soup.get_text(separator='\n', strip=True)
        return all_text

    except requests.exceptions.RequestException as e:
        return f"Error fetching the website: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage:
website_url = "https://www.example.com"  # Replace with the actual URL
text_content = get_website_text(website_url)
print(text_content)