import socket
import platform
from pathlib import Path

from psychopy import visual, core

if platform.system() == 'Linux':
    port = 4000
else:
    p = Path.home() / 'Documents' / '_bitronics_port.txt'
    port = int(p.read_text().strip())

images = sorted(Path('.').glob('*.png'))

print(f'Connecting to 127.0.0.1:{port}')
conn = socket.create_connection(('127.0.0.1', port))

win = visual.Window(units='height', size=(600, 800))

for i, image in enumerate(images, 1):
    im = visual.ImageStim(
        win,
        image=str(image),
    )
    w, h = im.size
    im.size = (w / h, 1)
    im.draw()
    win.flip()
    conn.send(f'{i}\n'.encode())
    core.wait(2)
    
conn.send(b'end\n')
core.wait(0.1)
conn.close()
core.quit()



