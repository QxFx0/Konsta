import logging
import json
import asyncio
import threading
import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from mitmproxy import http, ctx
from src.compression_engine import CompressionEngine
from src import api_parsers
from src.ca_manager import CAManager
from src.config import Config
from src.llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

class ContextCompressionProxy:
    """
    mitmproxy addon for intercepting HTTP/HTTPS requests to specific target hosts
    and applying context compression to the request body using both semantic compression
    and LLM-based context processing.
    """
    def __init__(self, target_hosts=None):
        # Initialize configuration
        self.config = Config()
        
        # target_hosts should be a list of domains, e.g., ['api.openai.com', 'api.anthropic.com']
        # Use provided target_hosts or fall back to config
        self.target_hosts = target_hosts or self.config.target_hosts
        logger.info(f"Initializing ContextCompressionProxy with target hosts: {self.target_hosts}")
        
        # Initialize core components
        self.compression_engine = CompressionEngine(self.config)
        self.llm_processor = LLMProcessor(self.config)
        self.ca_manager = CAManager(self.config)
        
        # Initialize dedicated event loop for async operations
        self._async_loop = None
        self._async_thread = None
        self._loop_started = False
        self._init_async_loop()
        
        # Initialize CA for HTTPS interception
        self._init_https_interception()
    
    def _init_async_loop(self):
        """Initialize a dedicated event loop in a separate thread for async operations."""
        def run_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            
            # Start the LLM processor
            self._async_loop.run_until_complete(self.llm_processor.start())
            self._loop_started = True
            
            # Keep the loop running
            self._async_loop.run_forever()
        
        self._async_thread = threading.Thread(target=run_loop, daemon=True)
        self._async_thread.start()
        
        # Wait for loop to be ready
        import time
        timeout = 5.0
        start = time.time()
        while not self._loop_started and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if not self._loop_started:
            logger.error("Failed to start async event loop")
    
    def _init_https_interception(self):
        """Initialize CA certificates for HTTPS interception."""
        try:
            # Ensure paths are Path objects to avoid AttributeError in ca_manager
            cert_path = Path(self.config.ca_cert_path)
            key_path = Path(self.config.ca_key_path)
            
            self.ca_manager.setup_ca(
                cert_path, 
                key_path
            )
            self.ca_manager.install_ca_system_wide(cert_path)
            logger.info("HTTPS interception initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HTTPS interception: {e}")
            # We don't raise here to allow the proxy to start even if CA setup fails
    
    def _dump_data(self, filename: str, data: Any):
        """Helper to dump raw data to the dumps directory."""
        try:
            dump_dir = Path("/home/liskil/Konsta/dumps")
            dump_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = dump_dir / f"{timestamp}_{filename}"
            
            with open(file_path, "w", encoding="utf-8") as f:
                if isinstance(data, (dict, list)):
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(data))
        except Exception as e:
            logger.error(f"Failed to dump data to {filename}: {e}")

    def is_target_request(self, host_or_flow) -> bool:
        """Check if the request is targeted at one of our configured hosts."""
        if isinstance(host_or_flow, http.HTTPFlow):
            host = host_or_flow.request.host
        else:
            host = host_or_flow
        
        return any(host.endswith(target) for target in self.target_hosts)
    
    def request(self, flow: http.HTTPFlow) -> None:
        """
        Intercept HTTP requests and apply context compression if the request
        is targeted at one of our configured hosts.
        """
        if not self.is_target_request(flow):
            return
        
        try:
            # DUMP: Original request
            self._dump_data(f"{flow.request.host}_original_req.json", {
                "url": flow.request.pretty_url,
                "headers": dict(flow.request.headers),
                "content": flow.request.get_text()
            })

            # CRITICAL: Skip Cerebras requests to avoid recursion/403
            if "cerebras.ai" in flow.request.host:
                logger.debug(f"Skipping Cerebras request to avoid recursion/403")
                return
            
            # Parse the request body
            if flow.request.content and len(flow.request.content) > 10 * 1024 * 1024:
                logger.warning(f"Request payload too large ({len(flow.request.content)} bytes). Skipping compression.")
                return

            request_body = flow.request.get_text()
            if not request_body:
                return
            
            # Parse the API request
            parsed_request = self._parse_request(request_body)
            if not parsed_request:
                return
            
            # Apply context compression
            compressed_request = self.compress_context(parsed_request, skip_llm=False)
            if compressed_request:
                # DUMP: Compressed request
                self._dump_data(f"{flow.request.host}_compressed_req.json", {
                    "content": compressed_request
                })
                
                # Update the request body with compressed context
                compressed_json = json.dumps(compressed_request)
                flow.request.set_text(compressed_json)
                logger.info(f"Compressed request for {flow.request.host}")
                logger.debug(f"Compressed payload size: {len(compressed_json)} bytes")
                logger.debug(f"Compressed payload keys: {list(compressed_request.keys())}")
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
    
    def response(self, flow: http.HTTPFlow) -> None:
        """
        Intercept HTTP responses and dump them for diagnostics.
        """
        try:
            # DUMP: Raw response
            self._dump_data(f"{flow.request.host}_response.json", {
                "url": flow.request.pretty_url,
                "status": flow.response.status_code,
                "headers": dict(flow.response.headers),
                "content": flow.response.get_text()
            })
        except Exception as e:
            logger.error(f"Error dumping response: {e}")
    
    def _parse_request(self, request_body: str) -> Optional[Dict[str, Any]]:
        """Parse the request body based on the API format."""
        try:
            data = json.loads(request_body)
            # Use the dispatcher function from api_parsers
            parsed_request = api_parsers.parse_request(data, self.config.proxy_host)
            
            if not parsed_request:
                return None
            
            # return a simplified dict for the compression engine
            return {
                "messages": [msg._asdict() for msg in parsed_request.messages],
                "provider": parsed_request.provider,
                "original_payload": parsed_request.payload
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse request body as JSON")
            return None
    
    def _parse_response(self, response_body: str) -> Optional[Dict[str, Any]]:
        """Parse the response body based on the API format."""
        try:
            data = json.loads(response_body)
            # Determine which parser to use based on the response structure
            if "choices" in data:
                return api_parsers.OpenAIParser.parse_response(data)
            elif "completion" in data:
                return api_parsers.AnthropicParser.parse_response(data)
            else:
                logger.warning(f"Unknown response format: {data}")
                return None
        except json.JSONDecodeError:
            logger.warning("Failed to parse response body as JSON")
            return None
    
    def compress_context(self, request_data: Dict[str, Any], skip_llm: bool = False) -> Optional[Dict[str, Any]]:
        """
        Apply context compression to the request data.
        
        Args:
            request_data: The request data to compress
            skip_llm: If True, skip LLM distillation (used for Cerebras requests to avoid recursion)
        """
        try:
            # Extract messages for compression
            messages = request_data.get("messages", [])
            provider = request_data.get("provider", "generic")
            original_payload = request_data.get("original_payload", {})

            # Step 1: Apply semantic compression (Local, Fast, Safe)
            compressed_messages = self.compression_engine.compress(messages)
            
            # Step 2: Apply LLM-based context processing (Remote, Slower)
            # Skip if explicitly requested (e.g., for Cerebras requests to avoid recursion)
            if not skip_llm and hasattr(self.llm_processor, 'process_context') and self._loop_started:
                try:
                    if compressed_messages and self._async_loop:
                        # Use the dedicated event loop instead of creating a new one
                        future = asyncio.run_coroutine_threadsafe(
                            asyncio.wait_for(
                                self.llm_processor.process_context(compressed_messages),
                                timeout=7.0
                            ),
                            self._async_loop
                        )
                        
                        try:
                            processed_messages = future.result(timeout=10.0)
                            if processed_messages:
                                compressed_messages = processed_messages
                                logger.info("LLM distillation applied successfully")
                            else:
                                logger.warning("LLM distillation returned no result")
                        except asyncio.TimeoutError:
                            logger.warning("LLM distillation timed out")
                        except Exception as e:
                            logger.warning(f"LLM Distillation failed: {e}")
                            
                except Exception as async_e:
                    logger.error(f"Async processing error: {async_e}. Using semantic compressed data.")
            elif skip_llm:
                logger.info("Skipping LLM distillation for Cerebras target to prevent 403/recursion")

            # Step 3: Reconstruct the original payload with compressed messages
            from src.api_parsers import serialize_request, ParsedRequest, Message
            
            standardized_msgs = [Message(role=m['role'], content=m['content']) for m in compressed_messages]
            
            parsed_obj = ParsedRequest(
                payload=original_payload,
                messages=standardized_msgs,
                provider=provider
            )
            
            final_payload = api_parsers.serialize_request(parsed_obj)
            logger.info(f"Final compressed payload prepared for provider: {provider}")
            return final_payload
        
        except Exception as e:
            logger.error(f"Error in compress_context: {e}")
            return None
    
    def decompress_context(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply context decompression to the response data.
        
        Currently not implemented as we don't modify responses.
        This is a placeholder for future functionality.
        """
        # Decompression not needed for current implementation
        return response_data

# Register the addon with mitmproxy
addons = [ContextCompressionProxy()]