import socket

host = socket.gethostname().lower()
if ('storm' in host) or ('braavos' in host):
    folder = '/storage/users/jack/diffusion_model/'
if ('exp' in host):
    folder = '/expanse/lustre/projects/cwr109/zhen1997/diffusion_model'
# print(folder)