from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code in [200, 503]


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200


def test_invalid_input():
    response = client.post(
        "/predict",
        json={"Item_Identifier": "FD001"}
    )
    assert response.status_code == 422


def test_batch_empty():
    response = client.post(
        "/batch_predict",
        json={"records": []}
    )
    assert response.status_code == 400


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "Item_Identifier": "FD001",
            "Item_Weight": 9.3,
            "Item_Fat_Content": "Low Fat",
            "Item_Visibility": 0.016,
            "Item_Type": "Dairy",
            "Item_MRP": 249.8,
            "Outlet_Identifier": "OUT049",
            "Outlet_Establishment_Year": 1999,
            "Outlet_Size": "Medium",
            "Outlet_Location_Type": "Tier 1",
            "Outlet_Type": "Supermarket Type1"
        }
    )

    assert response.status_code in [200, 503]