import requests
import math

LOCAL_URL = "http://127.0.0.1:8000"
DOCKER_URL = "http://127.0.0.1:8001"

sample_input = {
    "Item_Identifier": "FDA15",
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


def get_prediction(url):
    response = requests.post(
        f"{url}/predict",
        json=sample_input,
        timeout=30
    )

    response.raise_for_status()

    return response.json()["predicted_sales"]


def test_prediction_consistency():
    local_prediction = get_prediction(LOCAL_URL)
    docker_prediction = get_prediction(DOCKER_URL)

    print("Local prediction:", local_prediction)
    print("Docker prediction:", docker_prediction)

    assert math.isclose(
        local_prediction,
        docker_prediction,
        rel_tol=1e-5,
        abs_tol=0.01
    ), "Local and Docker predictions do not match!"

    print("PASS: Local and Docker predictions are consistent.")


if __name__ == "__main__":
    test_prediction_consistency()