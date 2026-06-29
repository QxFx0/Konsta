import sys
from unittest.mock import MagicMock, patch

# Mock mitmproxy before importing proxy_core
mitmproxy_mock = MagicMock()
sys.modules["mitmproxy"] = mitmproxy_mock
sys.modules["mitmproxy.http"] = mitmproxy_mock.http

from src.proxy_core import ContextCompressionProxy

def test_is_target_request_exact_match():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    
    # Exact match should pass
    flow = MagicMock()
    flow.request.pretty_host = "api.openai.com"
    assert proxy.is_target_request(flow) is True

def test_is_target_request_suffix_match():
    proxy = ContextCompressionProxy(target_hosts=["openai.com"])
    
    # Suffix match should pass
    flow = MagicMock()
    flow.request.pretty_host = "api.openai.com"
    assert proxy.is_target_request(flow) is True

def test_is_target_request_false_positive():
    proxy = ContextCompressionProxy(target_hosts=["openai.com"])
    
    # Substring match that is NOT a suffix match should fail
    flow = MagicMock()
    flow.request.pretty_host = "fake-api.openai.com.evil.com"
    assert proxy.is_target_request(flow) is False

def test_is_target_request_no_match():
    proxy = ContextCompressionProxy(target_hosts=["openai.com"])
    
    flow = MagicMock()
    flow.request.pretty_host = "google.com"
    assert proxy.is_target_request(flow) is False

def test_request_interception_target():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = MagicMock()
    flow.request.pretty_host = "api.openai.com"
    flow.request.content = b"test content"
    
    # Mock the compression logic to avoid actual network/complex calls
    with patch.object(proxy, 'compress_context', return_value=b"compressed") as mock_compress:
        proxy.request(flow)
        mock_compress.assert_called_once()
        assert flow.request.content == b"compressed"

def test_request_interception_non_target():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = MagicMock()
    flow.request.pretty_host = "google.com"
    flow.request.content = b"test content"
    
    with patch.object(proxy, 'compress_context') as mock_compress:
        proxy.request(flow)
        mock_compress.assert_not_called()
        assert flow.request.content == b"test content"

def test_response_interception_target():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = MagicMock()
    flow.request.pretty_host = "api.openai.com"
    flow.response.content = b"response content"
    
    with patch.object(proxy, 'decompress_context', return_value=b"decompressed") as mock_decompress:
        proxy.response(flow)
        mock_decompress.assert_called_once()
        assert flow.response.content == b"decompressed"

def test_response_interception_non_target():
    proxy = ContextCompressionProxy(target_hosts=["api.openai.com"])
    flow = MagicMock()
    flow.request.pretty_host = "google.com"
    flow.response.content = b"response content"
    
    with patch.object(proxy, 'decompress_context') as mock_decompress:
        proxy.response(flow)
        mock_decompress.assert_not_called()
        assert flow.response.content == b"response content"
