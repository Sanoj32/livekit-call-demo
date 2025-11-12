import os

from dotenv import load_dotenv

from livekit.plugins import openai
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt = openai.STT.with_azure(
            model="gpt-4o-transcribe",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-03-01-preview",
        ),
        llm=openai.LLM.with_azure(
            azure_deployment="gpt-4o",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # or AZURE_OPENAI_ENDPOINT
            api_key=os.getenv("AZURE_OPENAI_API_KEY"), # or AZURE_OPENAI_API_KEY
            api_version="2024-12-01-preview", # or OPENAI_API_VERSION
        ),
        tts=openai.TTS.with_azure(
            model="gpt-4o-mini-tts",
            voice="coral",
        ),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )



if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))