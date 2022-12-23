import os
from setuptools import setup


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "dwave", "inspector", "package_info.py")
package_info = {}
with open(package_info_path, encoding='utf-8') as f:
    exec(f.read(), package_info)


# Package requirements, minimal pinning
install_requires = [
    'dimod>=0.10.0',
    'dwave-system>=1.3.0',
    'dwave-cloud-client>=0.8.3',
    'Flask>=2.2',
    'numpy',
    # dwave-inspectorapp==0.3.1
]

# Package extras requirements
extras_require = {
    'test': ['coverage', 'vcrpy'],

    # importlib.resources backport needed for reading distribution data files
    ':python_version < "3.9"': ['importlib-resources>=3.2.0'],

    # importlib.metadata backport needed for entry point tests
    ':python_version < "3.10"': ['importlib-metadata>=4.8'],
}

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

packages = ['dwave', 'dwave.inspector']

python_requires = '>=3.7'

setup(
    name=package_info['__package_name__'],
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email=package_info['__author_email__'],
    description=package_info['__description__'],
    long_description=open('README.rst', encoding='utf-8').read(),
    url=package_info['__url__'],
    license=package_info['__license__'],
    packages=packages,
    entry_points={
        package_info['entry_point_group']['viewers']: [
            'jupyter_inline = dwave.inspector.viewers:jupyter_inline',
            'browser_tab = dwave.inspector.viewers:webbrowser_tab',
            'browser_window = dwave.inspector.viewers:webbrowser_window',
        ],
        package_info['entry_point_group']['proxies']: [
            'jupyter_server_proxy = dwave.inspector.proxies:jupyter_server_proxy',
        ],
        package_info['entry_point_group']['contrib']: [
            'dwave-inspector = dwave.inspector.package_info:contrib'
        ]
    },
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    zip_safe=False,
)
