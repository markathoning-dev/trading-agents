import typer
import uvicorn

app = typer.Typer()

@app.command()
def main(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
):
    uvicorn.run("web.main:app", host=host, port=port, reload=True)
