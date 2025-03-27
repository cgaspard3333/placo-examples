# Pressure sensors

To simulate pressure sensors, find the cylinder cleats, for example:

```xml
<geom size="0.008 0.008" pos="-0.0435 -0.0535 0.10947" quat="0 0 -0.707107 0.707107" type="cylinder" />
```

And wrap them in a separate body:

```xml
<body name="right_foot_cleat_0">
    <site name="right_foot_cleat_0" quat="0.5 -0.5 -0.5 -0.5" />

    <geom size="0.008 0.008" pos="0.0325 -0.0535 0.10947"
        quat="0 0 -0.707107 0.707107" type="cylinder" />
</body>
```

Here:
* The separate body allows for MuJoCo to handle the physics separately and get a separate force for each cleat
* The site is associated to the corresponding `<force>` sensor
* The site orientation is reflected from the `right_foot` site to ensure the orientation of the frame
* Force measured by MuJoCo is the force transmitted to the parent. In that case, it will be a negative upward
  force.

