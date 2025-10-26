from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Regex Assistant API", version="1.0")

class RegexRequest(BaseModel):
    prompt: str

class RegexResponse(BaseModel):
    regex: str
    explanation: str

@app.post("/regex-assistant", response_model=RegexResponse)
async def regex_assistant(req: RegexRequest):
    try:
        # Ask the LLM to generate regex + explanation
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful regex assistant. Respond with a regex and a short explanation."},
                {"role": "user", "content": req.prompt}
            ]
        )
        result = completion.choices[0].message.content.strip()

        # Simple parsing (LLM outputs regex + explanation separated by newline)
        parts = result.split("\n", 1)
        regex = parts[0].strip("`").replace("regex:", "").strip()
        explanation = parts[1].strip() if len(parts) > 1 else "No explanation provided."

        return RegexResponse(regex=regex, explanation=explanation)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
