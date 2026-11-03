from pydantic import BaseModel, Field
from typing import List


class BigMartInput(BaseModel):
    Item_Identifier: str
    Item_Weight: float = Field(gt=0)
    Item_Fat_Content: str
    Item_Visibility: float = Field(ge=0)
    Item_Type: str
    Item_MRP: float = Field(gt=0)
    Outlet_Identifier: str
    Outlet_Establishment_Year: int
    Outlet_Size: str
    Outlet_Location_Type: str
    Outlet_Type: str


class PredictionResponse(BaseModel):
    predicted_sales: float
    status: str


class BatchPredictionRequest(BaseModel):
    records: List[BigMartInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    total_records: int
    status: str