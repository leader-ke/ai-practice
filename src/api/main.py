import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import AgentConfigError, AgentError, AgentExecutionError, AgentTimeoutError, AgentUnavailableError, run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Checked in order, so a subclass must come before its base class.
# "expected" failures (the upstream is busy or slow) are logged as a warning without a traceback.
ERROR_RESPONSES = [
    (AgentConfigError, 500, "Internal server configuration error.", False),
    (AgentUnavailableError, 503, "Service temporarily unavailable. Try again later.", True),
    (AgentTimeoutError, 504, "Gateway timeout. Upstream service failed to respond.", True),
    (AgentExecutionError, 500, "An error occurred processing your request.", False),
    (AgentError, 500, "An internal agent process failed.", False),
]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        return AskResponse(answer=run_agent(request.question))
    except Exception as err:
        for error_type, status_code, detail, expected in ERROR_RESPONSES:
            if isinstance(err, error_type):
                if expected:
                    logger.warning("%s: %s", error_type.__name__, err)
                else:
                    logger.exception("%s", error_type.__name__)
                raise HTTPException(status_code=status_code, detail=detail) from err

        logger.exception("Unhandled application failure")
        raise HTTPException(status_code=500, detail="An unexpected system error occurred.") from err
