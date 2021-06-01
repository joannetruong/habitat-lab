class MagnumBindings < Formula
  desc "`Bindings for the Magnum C++11/C++14 graphics engine"
  homepage "https://magnum.graphics"
  url "https://github.com/mosra/magnum-bindings/archive/v2020.06.tar.gz"
  # wget https://github.com/mosra/magnum-bindings/archive/v2020.06.tar.gz -O - | sha256sum
  sha256 "959c703e6409ba0c2cd6c0da3a2b6190f6fac837ff69f64cbdc372e11359e7d8"
  head "git://github.com/mosra/magnum-bindings.git"

  depends_on "cmake"
  depends_on "python"
  depends_on "magnum"
  depends_on "pybind11" => :build

  # Apply a patch to make 2020.06 working with latest pybind11, which changes
  # py::module to py::module_
  stable do
    patch do
      url "https://github.com/mosra/magnum-bindings/commit/57db13422fe36d1bcfc7bee9b138c79901062dae.diff?full_index=1"
      sha256 "c13536e1bac8721f0c766768159e2b91babc4674fcbf46edfc0c43c72c77df3b"
    end
  end

  def install
    system "mkdir build"
    cd "build" do
      system "cmake",
        *std_cmake_args,
        "-DWITH_PYTHON=ON",
        ".."
      system "cmake", "--build", "."
      system "cmake", "--build", ".", "--target", "install"
      cd "src/python" do
        system "python3", *Language::Python.setup_install_args(prefix)
      end
    end
  end
end
