from fastapi import APIRouter
from controllers.todo_controller import router as todos_router
from controllers.ML_controller import router as ML_router

router = APIRouter()

router.include_router(todos_router)

router.include_router(ML_router)
