package com.example.circle.engine;
import com.example.circle.training.TrainingConfig.*;
import java.util.Random;
import java.util.function.BooleanSupplier;
import java.util.concurrent.CancellationException;

/** Primitive storage for conceptual {x,y,z}; targets always use the strict squared-distance rule. */
public final class CircleDataset {
    public enum Geometry { INSIDE, OUTSIDE, BOUNDARY }
    private final double[] x,y;
    private final byte[] target,geometry;
    private CircleDataset(int size) { x=new double[size]; y=new double[size]; target=new byte[size]; geometry=new byte[size]; }
    public int size(){return x.length;}
    public double x(int i){return x[i];} public double y(int i){return y[i];}
    public int target(int i){return target[i];}
    public Geometry geometry(int i){return Geometry.values()[geometry[i]];}
    public static int label(Circle c,double x,double y){double dx=x-c.x(),dy=y-c.y();return dx*dx+dy*dy<c.radius()*c.radius()?1:0;}
    public static Geometry geometry(Circle c,double x,double y){
        double distance=Math.hypot(x-c.x(),y-c.y());
        return Math.abs(distance-c.radius())<=c.tolerance()?Geometry.BOUNDARY:distance<c.radius()?Geometry.INSIDE:Geometry.OUTSIDE;
    }
    public record Split(CircleDataset training,CircleDataset validation,CircleDataset test) {}
    public static Split generate(Circle circle,Datasets config,BooleanSupplier cancelled) {
        int total=Math.addExact(Math.addExact(count(config.training()),count(config.validation())),count(config.test()));
        Keys used=new Keys(total);
        return new Split(generate(circle,config.training(),used,cancelled),generate(circle,config.validation(),used,cancelled),generate(circle,config.test(),used,cancelled));
    }
    public static int count(Dataset d) {
        if(d.strategy()!=Strategy.GRID)return d.sampleCount();
        long nx=(long)Math.floor((d.maxX()-d.minX())/d.step()+1e-9)+1;
        long ny=(long)Math.floor((d.maxY()-d.minY())/d.step()+1e-9)+1;
        if(nx<=0||ny<=0||nx>10_000_000||ny>10_000_000||nx*ny>10_000_000)throw new IllegalArgumentException("Grid exceeds 10 million points");
        return (int)(nx*ny);
    }
    private static CircleDataset generate(Circle c,Dataset d,Keys used,BooleanSupplier cancelled) {
        CircleDataset data=new CircleDataset(count(d)); Random random=new Random(d.seed());
        int nx=d.strategy()==Strategy.GRID?(int)Math.floor((d.maxX()-d.minX())/d.step()+1e-9)+1:0;
        long attempts=0;
        for(int i=0;i<data.size();) {
            if(cancelled.getAsBoolean())throw new CancellationException();
            if(++attempts>Math.max(10000L,100L*data.size()))throw new IllegalArgumentException("Cannot generate requested disjoint samples: enlarge bounds, reduce count or step, or change strategy");
            double x,y; boolean exact=i<d.exactBoundarySamples();
            if(exact) {
                // Construct boundary points explicitly. Correct a few rounding ULPs so that
                // the same strict squared-distance arithmetic used for labels equals r².
                double angle=random.nextDouble()*2*Math.PI;
                x=c.x()+c.radius()*Math.cos(angle); y=c.y()+c.radius()*Math.sin(angle);
                double radiusSquared=c.radius()*c.radius();
                for(int correction=0;correction<8;correction++) {
                    double dx=x-c.x(),dy=y-c.y(),distanceSquared=dx*dx+dy*dy;
                    if(distanceSquared==radiusSquared)break;
                    if(Math.abs(dx)>=Math.abs(dy))x=Math.nextAfter(x,distanceSquared<radiusSquared?(dx>=0?Double.POSITIVE_INFINITY:Double.NEGATIVE_INFINITY):c.x());
                    else y=Math.nextAfter(y,distanceSquared<radiusSquared?(dy>=0?Double.POSITIVE_INFINITY:Double.NEGATIVE_INFINITY):c.y());
                }
                double dx=x-c.x(),dy=y-c.y();
                if(dx*dx+dy*dy!=radiusSquared)continue;
            } else if(d.strategy()==Strategy.GRID) {x=d.minX()+(i%nx)*d.step(); y=d.minY()+(i/nx)*d.step();}
            else if(d.strategy()==Strategy.BOUNDARY_FOCUSED || (d.strategy()==Strategy.HYBRID && random.nextDouble()<d.boundaryFraction())) {
                double theta=random.nextDouble()*2*Math.PI;
                double offset=d.distribution()==Distribution.GAUSSIAN?random.nextGaussian()*d.boundaryWidth():(random.nextDouble()*2-1)*d.boundaryWidth();
                double r=c.radius()+offset;
                if(r<0)continue;
                x=c.x()+r*Math.cos(theta); y=c.y()+r*Math.sin(theta);
            } else {x=d.minX()+random.nextDouble()*(d.maxX()-d.minX());y=d.minY()+random.nextDouble()*(d.maxY()-d.minY());}
            if(!exact && d.step()>0 && d.strategy()!=Strategy.GRID) {
                x=d.minX()+Math.rint((x-d.minX())/d.step())*d.step(); y=d.minY()+Math.rint((y-d.minY())/d.step())*d.step();
            }
            if(x<d.minX()||x>d.maxX()||y<d.minY()||y>d.maxY())continue;
            if(!used.add(x,y)) {
                if(d.strategy()==Strategy.GRID)throw new IllegalArgumentException("Explicit boundary/grid samples overlap another dataset");
                continue;
            }
            data.x[i]=x;data.y[i]=y;data.target[i]=(byte)label(c,x,y);data.geometry[i]=(byte)geometry(c,x,y).ordinal();i++;
        }
        return data;
    }
    /** Primitive open-addressed fingerprints: collisions conservatively reject a point, never allow overlap. */
    private static final class Keys {
        final long[] keys;
        Keys(int count) {int capacity=16;while(capacity<count*1.5){if(capacity>=1<<27)throw new IllegalArgumentException("Combined datasets too large");capacity<<=1;}keys=new long[capacity];}
        boolean add(double x,double y) {
            long key=mix(Double.doubleToLongBits(x==0?0:x)) ^ Long.rotateLeft(mix(Double.doubleToLongBits(y==0?0:y)),29);
            if(key==0)key=1;
            int slot=(int)mix(key)&(keys.length-1);
            while(keys[slot]!=0) {if(keys[slot]==key)return false;slot=(slot+1)&(keys.length-1);}
            keys[slot]=key;return true;
        }
        static long mix(long v){v=(v^(v>>>30))*0xbf58476d1ce4e5b9L;v=(v^(v>>>27))*0x94d049bb133111ebL;return v^(v>>>31);}
    }
}
