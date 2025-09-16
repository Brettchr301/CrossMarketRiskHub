from __future__ import annotations


def test_api_contract_endpoints(api_client):
    paths = [
        "/v1/events/probabilities",
        "/v1/commodities/distributions",
        "/v1/companies/TNK/fundamental-state",
        "/v1/companies/TNK/valuation-distribution",
        "/v1/options/TNK/implied-distribution",
        "/v1/signals",
        "/v1/backtest/metrics",
    ]
    for path in paths:
        response = api_client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}: {response.text}"


def test_valuation_distribution_xlsx_export(api_client):
    response = api_client.get("/v1/companies/TNK/valuation-distribution.xlsx")
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    content_disposition = response.headers.get("content-disposition", "")
    assert "attachment;" in content_disposition
    assert "TNK_valuation_distribution.xlsx" in content_disposition
    assert len(response.content) > 2000
