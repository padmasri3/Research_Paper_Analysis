from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all analysis agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @abstractmethod
    def analyze(self, text: str, api_keys: Dict[str, str], *args, **kwargs) -> Dict[str, Any]:
        """Perform analysis on the given text."""
        pass
    
    def _make_api_request(self, url: str, headers: Dict[str, str] = None, 
                         params: Dict[str, Any] = None, json_data: Dict[str, Any] = None,
                         timeout: int = 30) -> Dict[str, Any]:
        """Make API request with error handling."""
        try:
            if json_data:
                response = self.session.post(url, headers=headers, json=json_data, timeout=timeout)
            else:
                response = self.session.get(url, headers=headers, params=params, timeout=timeout)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception(f"API request timed out after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Provide fallback analysis when APIs are unavailable."""
        return {"status": "fallback", "message": "Using local fallback method"}