import http.server
import socketserver
from http import HTTPStatus
from json import dumps

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        args = {}
        path=self.path
        if path.find("?") > 0:
            str=path.split("?")[1]
            for entry in str.split("&"):
                kv = entry.split("=")
                args[kv[0]] = kv[1]
        print(args)
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(dumps(args).encode())


httpd = socketserver.TCPServer(('', 8000), Handler)
httpd.serve_forever()
