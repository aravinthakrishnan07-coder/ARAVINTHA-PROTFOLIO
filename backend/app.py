from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Aravintha Krishnan R — Portfolio API",
    description="Standalone Python backend for portfolio metadata, OLS linear regression calculations, and internship inquiries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataPoint(BaseModel):
    x: float
    y: float


class RegressionRequest(BaseModel):
    points: List[DataPoint] = Field(..., min_length=2)


class RegressionPrediction(BaseModel):
    x: float
    actual_y: float
    predicted_y: float
    residual: float


class RegressionResponse(BaseModel):
    slope: float
    intercept: float
    mse: float
    r2: float
    equation: str
    predictions: List[RegressionPrediction]


class ContactMessageRequest(BaseModel):
    sender_name: str = Field(default="", max_length=120)
    sender_org: str = Field(default="", max_length=120)
    message: str = Field(..., min_length=1, max_length=3000)


class ContactMessageResponse(BaseModel):
    status: str
    received_at: str
    recipient_email: str
    preview: str


PORTFOLIO_DATA = {
    "profile": {
        "name": "Aravintha Krishnan R",
        "role": "B.Tech CSE (AI & ML) Student",
        "institution": "Jain University",
        "location": "Bengaluru, Karnataka",
        "email": "aravinthakrishnan07@gmail.com",
        "phone": "8838673603",
        "career_objective": (
            "Second-year B.Tech Computer Science Engineering student specializing in "
            "Artificial Intelligence and Machine Learning, seeking an internship opportunity "
            "to apply programming fundamentals, strengthen practical machine learning skills, "
            "and learn through real-world projects."
        ),
    },
    "skills": {
        "programming": ["Python", "C", "C++", "Java (Intermediate)"],
        "tools": ["Visual Studio Code"],
        "soft": [
            "Problem-solving",
            "Teamwork",
            "Adaptability",
            "Time Management",
            "Quick Learning",
        ],
        "languages": ["English", "Tamil", "Telugu"],
    },
    "education": [
        {
            "institution": "Jain University",
            "credential": "B.Tech, Computer Science Engineering (AI & ML)",
            "status": "2nd Year, 3rd Semester · Expected Graduation: 2029",
            "cgpa": "7.6",
        },
        {
            "institution": "Higher Secondary Education",
            "credential": "Class XII",
            "status": "Completed",
            "aggregate": "75%",
        },
        {
            "institution": "Secondary Education",
            "credential": "Class X",
            "status": "Completed",
            "aggregate": "75%",
        },
    ],
}

INQUIRY_LOG: List[dict] = []


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok", "runtime": "python"}


@app.get("/api/profile")
def get_profile() -> dict:
    return PORTFOLIO_DATA


@app.post("/api/regression", response_model=RegressionResponse)
def compute_linear_regression(payload: RegressionRequest) -> RegressionResponse:
    points = payload.points
    n = len(points)
    if n < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two data points are required to fit a regression line.",
        )

    mean_x = sum(p.x for p in points) / n
    mean_y = sum(p.y for p in points) / n

    numerator = sum((p.x - mean_x) * (p.y - mean_y) for p in points)
    denominator = sum((p.x - mean_x) ** 2 for p in points)

    slope = 0.0 if denominator == 0 else numerator / denominator
    intercept = mean_y - slope * mean_x

    predictions: List[RegressionPrediction] = []
    ss_res = 0.0
    ss_tot = 0.0

    for p in points:
        y_hat = slope * p.x + intercept
        residual = p.y - y_hat
        ss_res += residual ** 2
        ss_tot += (p.y - mean_y) ** 2
        predictions.append(
            RegressionPrediction(
                x=p.x,
                actual_y=p.y,
                predicted_y=round(y_hat, 4),
                residual=round(residual, 4),
            )
        )

    mse = ss_res / n
    r2 = 1.0 if ss_tot == 0 else max(0.0, 1.0 - (ss_res / ss_tot))

    safe_slope = 0.0 if abs(slope) < 0.005 else slope
    safe_intercept = 0.0 if abs(intercept) < 0.005 else intercept
    sign = "+" if safe_intercept >= 0 else "-"
    equation = f"y = {safe_slope:.2f}x {sign} {abs(safe_intercept):.2f}"

    return RegressionResponse(
        slope=round(slope, 6),
        intercept=round(intercept, 6),
        mse=round(mse, 6),
        r2=round(r2, 6),
        equation=equation,
        predictions=predictions,
    )


@app.post("/api/contact", response_model=ContactMessageResponse)
def submit_contact_inquiry(
    payload: ContactMessageRequest,
    sender_email: Optional[str] = None,
) -> ContactMessageResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "sender_name": payload.sender_name.strip() or "Hiring Team",
        "sender_org": payload.sender_org.strip(),
        "sender_email": sender_email,
        "message": message,
        "received_at": timestamp,
    }
    INQUIRY_LOG.append(entry)

    return ContactMessageResponse(
        status="received",
        received_at=timestamp,
        recipient_email=PORTFOLIO_DATA["profile"]["email"],
        preview=message[:140],
    )
