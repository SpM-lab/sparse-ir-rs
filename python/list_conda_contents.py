from conda_package_handling.api import handle_package
import sys

pkgfile = sys.argv[1]  # 例: ~/pkgs/mypkg-1.2.3-py38_0.conda
info = handle_package(pkgfile, "list")
for p in info:
    print(p)
EOF
