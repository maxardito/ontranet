
{
  description = "Development shell with LD_LIBRARY_PATH";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
      };
    in
      pkgs.mkShell {
        buildInputs = [
          pkgs.zlib
          pkgs.git-lfs
          pkgs.ffmpeg
        ];

        LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib.out}/lib";
      };
  };
}

