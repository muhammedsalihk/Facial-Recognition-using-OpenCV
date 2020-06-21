import time
tic = time.time()
import exec
exec.run()
tac = time.time()
delay = tac - tic
print(f'Recognised in {delay} seconds')