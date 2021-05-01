from setuptools import setup

package_name = 'robot_head'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shorton',
    maintainer_email='horton.rscotti@gmail.com',
    description='Robot Head Control Node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_head = robot_head.main:main',
        ],
    },
    scripts=['robot_head/pose.py']
)
