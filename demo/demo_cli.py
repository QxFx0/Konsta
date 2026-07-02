import asyncio

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

class KonstaClient:
    def __init__(self, proxy_url="http://127.0.0.1:8080"):
        self.proxy_url = proxy_url
        # We use a client that ignores SSL verification to work with Konsta's CA
        self.client = httpx.AsyncClient(
            proxy=self.proxy_url,
            verify=False,
            timeout=60.0
        )

    async def send_message(self, message: str, api_key: str):
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": message}]
        }

        try:
            response = await self.client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                # Extract compression metrics from headers
                orig_size = response.headers.get("X-Konsta-Original-Size", "0")
                comp_size = response.headers.get("X-Konsta-Compressed-Size", "0")

                metrics = ""
                if orig_size != "0" and comp_size != "0":
                    o, c = int(orig_size), int(comp_size)
                    savings = 100 - (c / o * 100) if o > 0 else 0
                    metrics = f"\n[bold green]📉 Compression: {o} → {c} chars ({savings:.1f}% reduction)[/bold green]"

                data = response.json()
                content = data['choices'][0]['message']['content']
                return f"{content}{metrics}"
            else:
                return f"[red]Error {response.status_code}: {response.text}[/red]"
        except Exception as e:
            return f"[red]Connection Error: {e}[/red]"

    async def close(self):
        await self.client.aclose()

async def main():
    console.clear()
    console.print(Panel("[bold magenta]KONSTA Client[/bold magenta] [bold white]Chat Interface[/bold white]", expand=False, border_style="magenta"))

    api_key = Prompt.ask("[bold yellow]Enter your Mistral API Key[/bold yellow]")

    client = KonstaClient()

    console.print("\n[dim]Chat started. Send messages to test compression. Type 'exit' or 'quit' to stop.[/dim]\n")

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input.strip():
            continue

        with console.status("[bold yellow]Waiting for compressed response...[/bold yellow]"):
            answer = await client.send_message(user_input, api_key)

        console.print(Panel(Markdown(answer), title="[bold magenta]Mistral[/bold magenta]", border_style="magenta"))
        console.print("\n")

    await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
