from app.api.routes import router
from app.api.execution_routes import router as execution_router

# Merge execution routes into main router
router.include_router(execution_router)

__all__ = ["router"]

