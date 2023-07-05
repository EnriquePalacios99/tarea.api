from fastapi import APIRouter, HTTPException, status
from models import Todo

router = APIRouter()

todos = []


@router.get("/todos")
def get_todos():
    return todos

@router.post('/todos', status_code=status.HTTP_201_CREATED)
def create_todo(todo : Todo):
    todos.append(todo)
    return {"message": "Creado satisfactoriamente"}

@router.put("/todos/{todo_id}")
def update_todo(todo_id: int , todo:Todo):
    for index,todo in enumerate(todos):
        if todo.id == todo_id:
            todo[index] = todo
            return todo
        
    raise HTTPException(status_code=404, detail='Not found')

@router.delete('/todos/{todo_id}')
def delete_todo(todo_id:int):
    for index,todo in enumerate(todos):
        if todo.id == todo_id:
            todos.remove(index)
            return {"message": "Borrado satisfactoriamente"}
        
    raise HTTPException(status_code=404, detail='Not found')