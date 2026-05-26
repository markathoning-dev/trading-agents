import os, sys
os.environ["KILO_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbnYiOiJwcm9kdWN0aW9uIiwia2lsb1VzZXJJZCI6IjhhZWYwNGQyLWM1NTUtNDc2Ny1iMzlhLTA0N2ZiMmRhMjQ3YyIsImFwaVRva2VuUGVwcGVyIjpudWxsLCJ2ZXJzaW9uIjozLCJpYXQiOjE3Nzk4MTY5OTYsImV4cCI6MTkzNzQ5Njk5Nn0.hF-aOcj4O9MiID5_ZSCABkqnoHyt6aQdwyvDiqYz41M"
os.environ["KILO_API_BASE"] = "https://api.kilo.ai/api/gateway"
os.environ["LITELLM_LOG"] = "ERROR"

sys.argv = ["cli.main", "backtest", "compare-risk", "--symbol", "AAPL", "--steps", "3600"]
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cli.main import app
app()
