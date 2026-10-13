import asyncio

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ragbase.database import setup_database
from ragbase.generation import generate
from ragbase.retrieval import retrieve

load_dotenv()

app = typer.Typer()
console = Console()


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask"),
    k: int = typer.Option(4, "--k", "-k", help="Number of chunks to retrieve", min=1, max=20),
):
    """Query the RAG pipeline interactively."""

    async def run_query():
        await setup_database()

        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

        with console.status("[bold green]Retrieving relevant chunks..."):
            chunks = await retrieve(question, k)

        console.print(f"[bold yellow]Retrieved {len(chunks)} chunks[/bold yellow]\n")

        for i, chunk in enumerate(chunks, 1):
            chunk_text = Text(f"[Source: {chunk.source}] (similarity: {chunk.similarity:.3f})\n{chunk.content}")
            console.print(Panel(chunk_text, title=f"Chunk {i}", border_style="blue"))

        with console.status("[bold green]Generating answer..."):
            response = await generate(question, chunks)

        console.print("\n[bold cyan]Answer:[/bold cyan]\n")
        console.print(Panel(response.answer, border_style="green"))

        console.print(f"\n[bold yellow]Sources:[/bold yellow] {', '.join(response.sources)}\n")

    asyncio.run(run_query())


if __name__ == "__main__":
    app()
