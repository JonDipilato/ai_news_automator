echo ''
echo -e '\033[90m#!/usr/bin/env python3\033[0m'
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
import sys
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
import json
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
from pathlib import Path
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
TODO_FILE = Path.home() / '.todo_cli.json'
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
def load_todos():
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
if TODO_FILE.exists():
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
return json.loads(TODO_FILE.read_text())
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
return []
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
def save_todos(todos):
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
TODO_FILE.write_text(json.dumps(todos, indent=2))
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
def cmd_add(args):
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
task = ' '.join(args)
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
if not task:
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print('Usage: todo add <task>')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
return
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
todos = load_todos()
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
todos.append({'task': task, 'done': False})
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
save_todos(todos)
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print(f'Added: {task}')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
def cmd_list():
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
todos = load_todos()
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
if not todos:
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print('No tasks yet. Add one with: todo add <task>')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
return
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
for i, todo in enumerate(todos, 1):
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
status = '✓' if todo['done'] else ' '
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print(f'{i}. [{status}] {todo["task"]}')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
def cmd_done(args):
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
todos = load_todos()
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
try:
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
idx = int(args[0]) - 1
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
todos[idx]['done'] = True
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
save_todos(todos)
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print(f'Marked as done: {todos[idx]["task"]}')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
except (IndexError, ValueError):
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print('Usage: todo done <number>')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
def main():
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
if len(sys.argv) < 2:
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print('Usage: todo <command> [args]')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
print('Commands: add, list, done')
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
return
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
cmd = sys.argv[1]
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
args = sys.argv[2:]
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
{'add': cmd_add, 'list': cmd_list, 'done': cmd_done}.get(cmd, lambda x: print(f'Unknown command: {cmd}'))(args)
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
if __name__ == '__main__':
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
main()
echo ''
echo -ne '\033[94m~\033[0m \033[92m$\033[0m '
