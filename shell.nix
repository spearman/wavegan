# $ nix-shell
# $ pipenv shell
# $ pipenv install
# $ ./train.sh

with import <nixpkgs> {};
mkShell {
  buildInputs = [
    pipenv
    (python3.withPackages (p: with p; [
      python-lsp-server
    ]))
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
