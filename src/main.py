import os
import sys
import subprocess
import logging
import signal
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed logging
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

def main():
    """
    Main entry point for the Konsta Context Compression Proxy.
    
    This function handles model selection, wires together the configuration,
    and launches the mitmproxy engine (mitmdump).
    """
    # Add project root to PYTHONPATH so mitmdump can find 'src' module
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    logger.info("Welcome to Konsta Context Compression Proxy")
    
    # 0. Model Selection (Simple CLI TUI)
    print("\n--- Select Compression Model (Cerebras.ai) ---")
    for key, model in config.SUPPORTED_MODELS.items():
        print(f"{key}) {model}")
    print("d) Use default/environment variable")
    
    choice = input("\nEnter choice [1-3 or d]: ").strip().lower()
    if choice in config.SUPPORTED_MODELS:
        selected_model = config.SUPPORTED_MODELS[choice]
        config.llm_model = selected_model
        logger.info(f"Selected model: {selected_model}")
    else:
        logger.info("Using default model configuration.")

    logger.info("Starting system...")
    
    # 1. Validate Environment and Config
    try:
        # Validate configuration first
        config._validate()
        
        subprocess.run(["mitmdump", "--version"], capture_output=True, check=True)
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("mitmdump not found in PATH. Please install mitmproxy: pip install mitmproxy")
        sys.exit(1)

    # 2. Prepare mitmdump command
    # We use mitmdump for a headless, production-ready proxy loop
    # -s loads the addon script where ContextCompressionProxy is defined
    # -p specifies the listening port
    
    addon_path = os.path.join(os.path.dirname(__file__), "proxy_core.py")
    
    # mitmproxy --set expects 'key=value' pairs. 
    # Using --set block_global=false ensures that we don't block global traffic 
    # and overrides any conflicting settings in the user's mitmproxy config files.
    cmd = [
        "mitmdump",
        "-s", addon_path,
        "-p", str(config.proxy_port),
        "--set", "block_global=false",
    ]

    logger.info(f"Launching proxy on {config.proxy_host}:{config.proxy_port}")
    logger.info(f"Targeting hosts: {', '.join(config.target_hosts)}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        # Start the proxy process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream logs from mitmdump to our logger
        for line in process.stdout:
            logger.info(f"[mitmproxy] {line.strip()}")

    except KeyboardInterrupt:
        logger.info("Shutting down proxy...")
        process.terminate()
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Handle termination signals for clean shutdown
    def signal_handler(sig, frame):
        logger.info("Received termination signal, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main()