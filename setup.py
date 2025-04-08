from setuptools import setup

package_name = 'sys_id_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ayush',
    maintainer_email='ayush18.kumar@gmail.com',
    description='Package for ontrack system identification using neural networks',
    license='Apache license 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sys_id = sys_id_py.on_track_sys_id:main'
        ],
    },
)
