import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import AgentConfigError, AgentError, AgentExecutionError, AgentTimeoutError, AgentUnavailableError, run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


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
        agent_response = run_agent(request.question)
        return AskResponse(answer=agent_response)
    except AgentConfigError as err:
        logger.exception("Configuration validation failure matching third-party API parameters")
        raise HTTPException(status_code=500, detail="Internal server configuration error.") from err

    except AgentUnavailableError as err:
        logger.exception("Upstream remote AI resource is temporarily unavailable")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Try again later.") from err

    except AgentTimeoutError as err:
        logger.exception("Upstream gateway execution limit reached")
        raise HTTPException(status_code=504, detail="Gateway timeout. Upstream service failed to respond.") from err

    except AgentExecutionError as err:
        logger.exception("Controlled agent loop runtime boundary error")
        raise HTTPException(status_code=500, detail="An error occurred processing your request.") from err

    except AgentError as err:
        logger.exception("General agent error intercepted")
        raise HTTPException(status_code=500, detail="An internal agent process failed.") from err

    except Exception as err:
        logger.exception("Unhandled application domain failure")
        raise HTTPException(status_code=500, detail="An unexpected system error occurred.") from err
