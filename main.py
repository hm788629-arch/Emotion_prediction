

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import pickle
import re
import traceback
from pathlib import Path


# 1. Configuration
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "Artifact" / "brigru_model.keras"
tokenizer_path = BASE_DIR / "Artifact" / "tokenizer.pkl"
static_path = BASE_DIR / "static"

max_sequence_length = 100

emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
    
]

emotion_emoji = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# 2. Text preprocessing
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 3. Request and response schemas
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# 4. Model loading and lifespan management
dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")

    try:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}"
            )

        dl_model["BRIGRU"] = load_model(
    str(model_path),
    compile=False
)

        with open(tokenizer_path, "rb") as file:
            dl_model["Tokenizer"] = pickle.load(file)

        print("Model and tokenizer loaded successfully!")

        yield

    finally:
        dl_model.clear()
        print("Application resources cleaned up.")


# 5. FastAPI application
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(static_path)),
    name="static"
)


# 6. API endpoints
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse(str(static_path / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health_check():
    loaded = (
        dl_model.get("BRIGRU") is not None
        and dl_model.get("Tokenizer") is not None
    )

    return HealthResponse(
        status="Server is running",
        model_loaded=loaded
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    try:
        bigru_model = dl_model.get("BRIGRU")
        tokenizer_model = dl_model.get("Tokenizer")

        if bigru_model is None or tokenizer_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model or tokenizer is not loaded."
            )

        cleaned_text = preprocess_text(text_input.text)

        if not cleaned_text:
            raise HTTPException(
                status_code=422,
                detail="Please enter valid text."
            )

        tokenized_text = tokenizer_model.texts_to_sequences(
            [cleaned_text]
        )

        padded_sequence = pad_sequences(
            tokenized_text,
            maxlen=max_sequence_length,
            padding="post",
            truncating="post"
        )

        print("Input shape:", padded_sequence.shape)
        print("Model input shape:", bigru_model.input_shape)
        print("Model output shape:", bigru_model.output_shape)

        probabilities = bigru_model.predict(
            padded_sequence,
            verbose=0
        )[0]

        print("Prediction probabilities:", probabilities)

        if len(probabilities) != len(emotion_labels):
            raise ValueError(
                f"Model returned {len(probabilities)} outputs, "
                f"but there are {len(emotion_labels)} emotion labels."
            )

        top_index = int(np.argmax(probabilities))
        all_probabilities = {
            label: float(probability)
            for label, probability in zip(
                emotion_labels, probabilities
            )
        }

        return PredictionResponse(
            text=text_input.text,
            predicted_emotion=emotion_labels[top_index],
            confidence=float(probabilities[top_index]),
            all_probabilities=all_probabilities
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}"
        ) from exc

