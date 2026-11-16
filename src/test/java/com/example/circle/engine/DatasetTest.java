package com.example.circle.engine;
import com.example.circle.training.TrainingConfig;
import com.example.circle.training.TrainingConfig.*;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
class DatasetTest {
    final Circle circle=new Circle(0,0,5,.01);
    Dataset spec(int count,long seed,Strategy strategy,double step,int boundary){return new Dataset(count,seed,-7,7,-7,7,strategy,step,.5,.1,Distribution.GAUSSIAN,boundary);}
    CircleDataset.Split split(Dataset a){return CircleDataset.generate(circle,new Datasets(a,spec(200,2,Strategy.UNIFORM,0,0),spec(200,3,Strategy.UNIFORM,0,0)),()->false);}
    @Test void strictLabelsAndDiagnosticTolerance(){
        assertEquals(1,CircleDataset.label(circle,0,0));assertEquals(0,CircleDataset.label(circle,3,4));assertEquals(0,CircleDataset.label(circle,5,0));
        assertEquals(CircleDataset.Geometry.BOUNDARY,CircleDataset.geometry(circle,4.999,0));assertEquals(1,CircleDataset.label(circle,4.999,0));
        assertEquals(CircleDataset.Geometry.OUTSIDE,CircleDataset.geometry(circle,6,0));assertEquals(CircleDataset.Geometry.INSIDE,CircleDataset.geometry(circle,0,0));
        var data=split(spec(100,1,Strategy.UNIFORM,0,100)).training();for(int i=0;i<100;i++){assertEquals(0,data.target(i));assertEquals(CircleDataset.Geometry.BOUNDARY,data.geometry(i));}
    }
    @Test void reproducibleDisjointAndQuantized(){
        var a=split(spec(1000,1,Strategy.HYBRID,.01,0));var b=split(spec(1000,1,Strategy.HYBRID,.01,0));Set<String> coordinates=new HashSet<>();
        for(var data:List.of(a.training(),a.validation(),a.test()))for(int i=0;i<data.size();i++){
            assertTrue(coordinates.add(data.x(i)+","+data.y(i)));assertTrue(data.x(i)>=-7&&data.x(i)<=7&&data.y(i)>=-7&&data.y(i)<=7);assertEquals(CircleDataset.label(circle,data.x(i),data.y(i)),data.target(i));
        }
        for(int i=0;i<a.training().size();i++){
            assertEquals(a.training().x(i),b.training().x(i));assertEquals(a.training().y(i),b.training().y(i));
            assertEquals(Math.rint((a.training().x(i)+7)/.01),(a.training().x(i)+7)/.01,1e-9);
            assertEquals(Math.rint((a.training().y(i)+7)/.01),(a.training().y(i)+7)/.01,1e-9);
        }
    }
    @Test void distributionsAndCounts(){
        double[] near=new double[3];int index=0;
        for(Strategy strategy:List.of(Strategy.UNIFORM,Strategy.BOUNDARY_FOCUSED,Strategy.HYBRID)){
            CircleDataset data=split(spec(10000,1,strategy,0,0)).training();assertEquals(10000,data.size());double xMean=0;int positive=0;
            for(int i=0;i<data.size();i++){xMean+=data.x(i);if(data.x(i)>0)positive++;if(Math.abs(Math.hypot(data.x(i),data.y(i))-5)<.3)near[index]++;}
            assertEquals(0,xMean/data.size(),.15);assertEquals(.5,positive/(double)data.size(),.03);index++;
        }
        assertTrue(near[1]>9500);assertTrue(near[0]<2000);assertTrue(near[2]>4500&&near[2]<6500);
    }
    @Test void gridSizeNotTruncatedAndOverlapFails(){
        Dataset grid=spec(1,1,Strategy.GRID,1,0);assertEquals(225,split(grid).training().size());
        assertThrows(IllegalArgumentException.class,()->CircleDataset.generate(circle,new Datasets(grid,grid,grid),()->false));
        assertThrows(IllegalArgumentException.class,()->split(spec(1000,1,Strategy.UNIFORM,14,0)));
    }
    @Test void millionSamplesUsePrimitiveStorage(){assertEquals(1_000_000,split(spec(1_000_000,1,Strategy.UNIFORM,0,0)).training().size());}
}
